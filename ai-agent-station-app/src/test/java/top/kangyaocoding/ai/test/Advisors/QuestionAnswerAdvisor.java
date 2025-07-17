package top.kangyaocoding.ai.test.Advisors;

import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain;
import org.springframework.ai.chat.client.advisor.api.StreamAdvisorChain;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionTextParser;
import org.springframework.lang.Nullable;
import org.springframework.util.Assert;
import org.springframework.util.StringUtils;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 描述: RAG 风格的 Advisor，从 Vector Store 中检索上下文信息并注入到用户提示中。
 * 参考官方 1.1.0-SNAPSHOT QuestionAnswerAdvisor 的源码
 *
 * @author K·Herbert herbert501@qq.com
 * @since 2025-07-17 09:40
 */
public class QuestionAnswerAdvisor implements BaseAdvisor {

    public static final String QA_RETRIEVED_DOCUMENTS_KEY = "qa_retrieved_documents";
    public static final String QA_FILTER_EXPRESSION_KEY = "qa_filter_expression";

    private static final PromptTemplate DEFAULT_PROMPT_TEMPLATE = new PromptTemplate("""
            {query}
            
            Context information is below, surrounded by ---------------------
            
            ---------------------
            {question_answer_context}
            ---------------------
            
            Given the context and provided history information and not prior knowledge,
            reply to the user comment. If the answer is not in the context, inform
            the user that you can't answer the question.
            """);

    private static final int DEFAULT_ORDER = 0;

    private final VectorStore vectorStore;
    private final SearchRequest searchRequest;
    private final PromptTemplate promptTemplate;
    private final Scheduler scheduler;
    private final int order;

    public QuestionAnswerAdvisor(VectorStore vectorStore) {
        this(vectorStore, SearchRequest.builder().build());
    }

    public QuestionAnswerAdvisor(VectorStore vectorStore, SearchRequest searchRequest) {
        this(vectorStore, searchRequest, null, null, DEFAULT_ORDER);
    }

    QuestionAnswerAdvisor(VectorStore vectorStore, SearchRequest searchRequest,
                          @Nullable PromptTemplate promptTemplate,
                          @Nullable Scheduler scheduler, int order) {
        Assert.notNull(vectorStore, "VectorStore must not be null");
        Assert.notNull(searchRequest, "SearchRequest must not be null");

        this.vectorStore = vectorStore;
        this.searchRequest = searchRequest;
        this.promptTemplate = promptTemplate != null ? promptTemplate : DEFAULT_PROMPT_TEMPLATE;
        this.scheduler = scheduler != null ? scheduler : Schedulers.boundedElastic();
        this.order = order;
    }

    public static Builder builder(VectorStore vectorStore) {
        return new Builder(vectorStore);
    }

    @Override
    public ChatClientResponse adviseCall(ChatClientRequest chatClientRequest, CallAdvisorChain callAdvisorChain) {
        return callAdvisorChain.nextCall(this.before(chatClientRequest, callAdvisorChain));
    }

    @Override
    public Flux<ChatClientResponse> adviseStream(ChatClientRequest chatClientRequest, StreamAdvisorChain streamAdvisorChain) {
        return BaseAdvisor.super.adviseStream(chatClientRequest, streamAdvisorChain);
    }

    @Override
    public String getName() {
        return this.getClass().getSimpleName();
    }

    @Override
    public ChatClientRequest before(ChatClientRequest chatClientRequest, AdvisorChain advisorChain) {
        // 1. 创建查询文本
        String query = chatClientRequest.prompt().getUserMessage().getText();

        // 2. 构建搜索请求
        SearchRequest searchRequestToUse = SearchRequest.from(this.searchRequest)
                .query(query)
                .filterExpression(this.doGetFilterExpression(chatClientRequest.context()))
                .build();
        // 3. 检索知识库
        List<Document> documents = this.vectorStore.similaritySearch(searchRequestToUse);
        // 4. 构建上下文
        Map<String, Object> context = new HashMap<>(chatClientRequest.context());
        context.put(QA_RETRIEVED_DOCUMENTS_KEY, documents);

        // 5. 拼接文档的内容
        String documentContext = documents == null || documents.isEmpty() ? "" :
                documents.stream()
                        .map(Document::getText)
                        .collect(Collectors.joining(System.lineSeparator()));
        // 6. 构建增强后的用户提示词
        UserMessage originalUserMessage = chatClientRequest.prompt().getUserMessage();
        String augmentedUserText = this.promptTemplate.render(Map.of(
                "query", originalUserMessage.getText(),
                "question_answer_context", documentContext
        ));
        // 7. 构建新的 ChatClientRequest
        return chatClientRequest.mutate()
                .prompt(chatClientRequest.prompt().augmentUserMessage(augmentedUserText))
                .context(context)
                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse chatClientResponse, AdvisorChain advisorChain) {

        ChatResponse.Builder chatResponseBuilder = chatClientResponse.chatResponse() == null ?
                ChatResponse.builder() :
                ChatResponse.builder().from(chatClientResponse.chatResponse());
        chatResponseBuilder.metadata(QA_RETRIEVED_DOCUMENTS_KEY, chatClientResponse.context().get(QA_RETRIEVED_DOCUMENTS_KEY));

        return ChatClientResponse.builder()
                .chatResponse(chatResponseBuilder.build())
                .context(chatClientResponse.context())
                .build();
    }

    @Override
    public Scheduler getScheduler() {
        return this.scheduler;
    }

    @Override
    public int getOrder() {
        return this.order;
    }

    @Nullable
    protected Filter.Expression doGetFilterExpression(Map<String, Object> context) {
        if (context.containsKey(QA_FILTER_EXPRESSION_KEY) &&
                StringUtils.hasText(context.get(QA_FILTER_EXPRESSION_KEY).toString())) {
            return new FilterExpressionTextParser().parse(context.get(QA_FILTER_EXPRESSION_KEY).toString());
        }
        return this.searchRequest.getFilterExpression();
    }

    // Builder 模式
    public static final class Builder {
        private final VectorStore vectorStore;
        private SearchRequest searchRequest = SearchRequest.builder().build();
        private PromptTemplate promptTemplate;
        private Scheduler scheduler;
        private int order = DEFAULT_ORDER;

        private Builder(VectorStore vectorStore) {
            Assert.notNull(vectorStore, "VectorStore must not be null");
            this.vectorStore = vectorStore;
        }

        public Builder promptTemplate(PromptTemplate promptTemplate) {
            this.promptTemplate = promptTemplate;
            return this;
        }

        public Builder searchRequest(SearchRequest searchRequest) {
            this.searchRequest = searchRequest;
            return this;
        }

        public Builder protectFromBlocking(boolean protectFromBlocking) {
            this.scheduler = protectFromBlocking ? Schedulers.boundedElastic() : Schedulers.immediate();
            return this;
        }

        public Builder scheduler(Scheduler scheduler) {
            this.scheduler = scheduler;
            return this;
        }

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public QuestionAnswerAdvisor build() {
            return new QuestionAnswerAdvisor(vectorStore, searchRequest, promptTemplate, scheduler, order);
        }
    }
}
