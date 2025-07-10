package top.kangyaocoding.ai.test;

import com.alibaba.fastjson2.JSON;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.ai.content.Media;
import org.springframework.ai.document.Document;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.util.MimeType;
import org.springframework.util.MimeTypeUtils;
import reactor.core.publisher.Flux;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 描述: OpenAi 对话测试类
 *
 * @author K·Herbert herbert501@qq.com
 * @since 2025-07-09 09:26
 */
@Slf4j
@SpringBootTest
@RunWith(SpringRunner.class)
public class OpenAiTest {
    @Resource
    private OpenAiChatModel openAiChatModel;
    @Resource
    private PgVectorStore pgVectorStore;
    @Resource
    private TokenTextSplitter tokenTextSplitter;

    @Test
    public void testCall() {
        log.info("测试call服务");
        ChatResponse chatResponse = openAiChatModel.call(new Prompt("你是谁？",
                OpenAiChatOptions.builder()
                        .model("deepseek-v3")
                        .maxCompletionTokens(1024)
                        .temperature(0.7)
                        .build()
        ));
        log.info("结果：{}", JSON.toJSONString(chatResponse.getResults()));
    }

    @Test
    public void testStream() {
        log.info("测试stream服务");
        Flux<ChatResponse> chatResponse = openAiChatModel.stream(new Prompt("你是谁？",
                OpenAiChatOptions.builder()
                        .model("deepseek-v3")
                        .maxCompletionTokens(1024)
                        .temperature(0.7)
                        .build()
        ));
        chatResponse.doOnNext(response ->
                        log.info("结果：{}", JSON.toJSONString(response.getResults())))
                .blockLast();

    }

    @Test
    public void testImage() throws URISyntaxException {
        log.info("测试image服务");
        UserMessage userMessage = UserMessage.builder()
                .text("请用中文描述这个图片")
                .media(Media.builder()
                        .mimeType(MimeType.valueOf(MimeTypeUtils.IMAGE_PNG_VALUE))
                        .data(new URI("https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AAOEcgc.img"))
                        .build())
                .build();
        ChatResponse chatResponse = openAiChatModel.call(new Prompt(userMessage,
                OpenAiChatOptions.builder()
                        .model("qwen-vl-max-2025-04-08")
                        .maxCompletionTokens(1024)
                        .temperature(0.7)
                        .build()));
        log.info("结果：{}", JSON.toJSONString(chatResponse.getResults()));
    }

    @Test
    public void testUploadFile() {
        log.info("测试upload file服务");
        TikaDocumentReader tikaDocumentReader = new TikaDocumentReader("static/prompt.txt");
        List<Document> documents = tikaDocumentReader.read();
        List<Document> splitDocuments = tokenTextSplitter.apply(documents);

        splitDocuments.forEach(document -> {
            log.info("文档：{}", document);
            document.getMetadata().put("knowledge", "article-prompt-words");
        });

        pgVectorStore.accept(splitDocuments);

        log.info("向量库保存成功");
    }

    @Test
    public void testKnowledge() {
        log.info("测试knowledge服务");
        // 系统提示模板，要求模型基于检索文档回答但表现得像已知信息
        String SYSTEM_PROMPT = """
                请严格依据提供的 DOCUMENTS 内容进行回答，确保答案与文档信息完全一致。
                                如果在文档中未找到相关信息，请直接回复“未找到相关信息”。
                                回答必须使用中文，并保持自然流畅的口语化表达。
                                DOCUMENTS：
                                    {documents}
                """;

        SearchRequest searchRequest = SearchRequest.builder()
                .query("王大瓜的个人信息是什么？")
                .topK(5)
                .filterExpression("knowledge === '王大瓜知识库'")
                .build();

        List<Document> documents = pgVectorStore.similaritySearch(searchRequest);
        log.info("搜索结果：{}", documents);

        if (documents.isEmpty() || documents == null) {
            log.info("没有搜索结果");
            Document document = Document.builder()
                    .text("没有搜索结果，直接输出未找到相关信息")
                    .build();
            documents.add(document);
        }

        String documentsCollection = documents.stream()
                .map(Document::getText)
                .collect(Collectors.joining("\n"));

        Message ragMessage = new SystemPromptTemplate(SYSTEM_PROMPT)
                .createMessage(Map.of("documents", documentsCollection));

        List<Message> messages = new ArrayList<>();
        messages.add(new UserMessage("王大瓜的个人信息是什么？"));
        messages.add(ragMessage);
        ChatResponse chatResponse = openAiChatModel.call(new Prompt(messages,
                OpenAiChatOptions.builder()
                        .model("qwen-turbo-2025-04-28")
                        .maxCompletionTokens(1024)
                        .temperature(0.7)
                        .build()));
        log.info("结果：{}", JSON.toJSONString(chatResponse.getResults()));
    }



}
