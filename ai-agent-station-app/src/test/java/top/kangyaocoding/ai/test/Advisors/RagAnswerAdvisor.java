package top.kangyaocoding.ai.test.Advisors;

import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisorChain;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionTextParser;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.util.StringUtils;
import reactor.core.publisher.Flux;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class RagAnswerAdvisor implements BaseAdvisor {

    private final PgVectorStore vectorStore;
    private final SearchRequest searchRequest;

    // 提示模板：包含 {question_answer_context} 占位符
    private final String userTextAdvise = """
        Context information is below, surrounded by ---------------------
        
        ---------------------
        {question_answer_context}
        ---------------------

        Given the context and provided history information and not prior knowledge,
        reply to the user comment. If the answer is not in the context, inform
        the user that you can't answer the question.
        """;

    public RagAnswerAdvisor(PgVectorStore vectorStore, SearchRequest searchRequest) {
        this.vectorStore = vectorStore;
        this.searchRequest = searchRequest;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest chatClientRequest, AdvisorChain advisorChain) {
        Map<String, Object> context = new HashMap<>(chatClientRequest.context());

        // 获取用户原始输入
        String userText = chatClientRequest.prompt().getUserMessage().getText();

        // 使用用户输入构造查询语句
        String query = (new PromptTemplate(userText)).render();

        // 构造带 filter 的搜索请求
        SearchRequest searchRequestToUse = SearchRequest.from(this.searchRequest)
                .query(query)
                .filterExpression(this.doGetFilterExpression(context))
                .build();

        // 执行向量检索
        List<Document> documents = this.vectorStore.similaritySearch(searchRequestToUse);

        // 将检索结果保存到上下文中供后续处理使用
        context.put("qa_retrieved_documents", documents);

        // 构造上下文内容
        String documentContext = documents.stream()
                .map(Document::getText)
                .collect(Collectors.joining(System.lineSeparator()));

        // 替换模板中的占位符
        String promptWithRag = this.userTextAdvise.replace("{question_answer_context}", documentContext);

        // 拼接到最终输入中
        String combinedInput = userText + System.lineSeparator() + promptWithRag;

        // 构建新的 Prompt（只包含 UserMessage）
        Prompt newPrompt = Prompt.builder()
                .messages(new UserMessage(combinedInput))
                .build();

        return ChatClientRequest.builder()
                .prompt(newPrompt)
                .context(context)
                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse chatClientResponse, AdvisorChain advisorChain) {
        ChatResponse.Builder chatResponseBuilder = ChatResponse.builder().from(chatClientResponse.chatResponse());
        chatResponseBuilder.metadata("qa_retrieved_documents", chatClientResponse.context().get("qa_retrieved_documents"));
        ChatResponse chatResponse = chatResponseBuilder.build();

        return ChatClientResponse.builder()
                .chatResponse(chatResponse)
                .context(chatClientResponse.context())
                .build();
    }

    @Override
    public ChatClientResponse adviseCall(ChatClientRequest chatClientRequest, CallAdvisorChain callAdvisorChain) {
        ChatClientResponse response = callAdvisorChain.nextCall(this.before(chatClientRequest, callAdvisorChain));
        return this.after(response, callAdvisorChain);
    }

    @Override
    public int getOrder() {
        return 0;
    }

    @Override
    public String getName() {
        return this.getClass().getSimpleName();
    }

    protected Filter.Expression doGetFilterExpression(Map<String, Object> context) {
        if (context.containsKey("qa_filter_expression") && StringUtils.hasText(context.get("qa_filter_expression").toString())) {
            return new FilterExpressionTextParser().parse(context.get("qa_filter_expression").toString());
        }
        return this.searchRequest.getFilterExpression();
    }

    // 如果需要支持流式输出，请取消注释并实现该方法
    @Override
    public Flux<ChatClientResponse> adviseStream(ChatClientRequest chatClientRequest, StreamAdvisorChain streamAdvisorChain) {
        ChatClientRequest modifiedRequest = before(chatClientRequest, streamAdvisorChain);
        return streamAdvisorChain.nextStream(modifiedRequest)
                .map(response -> after(response, streamAdvisorChain));
    }
}
