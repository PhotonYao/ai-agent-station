package top.kangyaocoding.ai.test;

import com.alibaba.fastjson.JSON;
import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.ServerParameters;
import io.modelcontextprotocol.client.transport.StdioClientTransport;
import io.modelcontextprotocol.spec.McpSchema;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.PromptChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;
import reactor.core.publisher.Flux;
import top.kangyaocoding.ai.test.Advisors.RagAnswerAdvisor;

import java.time.Duration;
import java.time.LocalDate;
import java.util.HashMap;
import java.util.Map;

/**
 * 描述: 智能体测试类
 *
 * @author K·Herbert herbert501@qq.com
 * @since 2025-07-09 15:05
 */
@Slf4j
@SpringBootTest
@RunWith(SpringRunner.class)
public class AiAgentTest {
    private ChatModel chatModel;
    private ChatClient chatClient;
    @Resource
    private PgVectorStore pgVectorStore;
    @Value("${spring.ai.openai.api-key}")
    private String apiKey;
    @Value("${spring.ai.openai.base-url}")
    private String baseUrl;

    public static final String CHAT_MEMORY_CONVERSATION_ID_KEY = "chat_memory_conversation_id";
    public static final String CHAT_MEMORY_RETRIEVE_SIZE_KEY = "chat_memory_response_size";

    @Before
    public void init() {
        OpenAiApi openAiApi = OpenAiApi.builder()
                .baseUrl(baseUrl)
                .apiKey(apiKey)
                .completionsPath("/v1/chat/completions")
                .embeddingsPath("/v1/embeddings")
                .build();

        chatModel = OpenAiChatModel.builder()
                .openAiApi(openAiApi)
                .defaultOptions(OpenAiChatOptions.builder()
                        .model("qwen-max-2025-01-25")
                        .toolCallbacks(new SyncMcpToolCallbackProvider(
                                fileSystemMcpClient(), mcpDingdingBotClient()).getToolCallbacks())
                        .build())
                .build();
        chatClient = ChatClient.builder(chatModel)
                .defaultSystem("""
                            你是一个 AI Agent 智能体，可以根据用户输入的信息自动生成 Markdown 技术文章，并通过文件系统工具保存，最后使用钉钉机器人工具JavaSDKMCPClient_send_text_message推送通知。今天是 {current_date}。
                        
                            你擅长使用 Planning 模式来分步骤完成任务，具体流程如下：
                        
                            1. 分析用户的输入内容，理解需求并生成结构化的 Markdown 技术文章；
                            2. 使用文件系统工具创建md文件，并将文章写入md该文件；保存文件路径为：E:/桌面文件/记事本/mcp_file_system；
                            3. 提取以下结构化信息：
                               - 文章标题（需包含技术点）
                               - 文章标签（多个用英文逗号隔开）
                               - 文章简述（不超过 100 字）
                            4. 使用钉钉机器人工具将文章标题、简述及保存路径作为纯文本消息发送出去；
                        
                            请根据以上规则自动规划任务流程，并调用相应的工具完成操作。
                        """)
                .defaultToolCallbacks(new SyncMcpToolCallbackProvider(fileSystemMcpClient(), mcpDingdingBotClient()).getToolCallbacks())
                .defaultAdvisors(
                        PromptChatMemoryAdvisor.builder(
                                MessageWindowChatMemory.builder()
                                        .maxMessages(10)
                                        .build()
                        ).build(),

                        // TODO 如果去掉 new RagAnswerAdvisor 这部分就可以正常处理整个流程
                        new RagAnswerAdvisor(pgVectorStore, SearchRequest.builder()
                                .topK(5)
                                .filterExpression("knowledge == '知识库名称-v4'")
                                .build()),
                        SimpleLoggerAdvisor.builder().build()
                ).build();
    }

    @Test
    public void test_chat_client_call() {
        String content = chatClient.prompt("使用文件系统工具，创建一个1.txt文件到 E:/桌面文件/记事本/mcp_file_system 的目录下，写入测试两个字").call().content();
        log.info("content: {}", content);
    }

    @Test
    public void test_chat_model_call() {
        Prompt prompt = Prompt.builder()
                .messages(new UserMessage(
                        """
                                    你是一个 AI Agent 智能体，可以根据用户输入的信息自动生成 Markdown 技术文章，并通过文件系统工具保存，最后使用钉钉机器人工具JavaSDKMCPClient_send_text_message推送通知。今天是 {current_date}。
                                
                                    你擅长使用 Planning 模式来分步骤完成任务，具体流程如下：
                                
                                    1. 分析用户的输入内容，理解需求并生成结构化的 Markdown 技术文章；
                                    2. 使用文件系统工具，保存文件路径为：E:/桌面文件/记事本/mcp_file_system；创建.md文件，并将文章内容写入该文件；
                                    3. 提取以下结构化信息：
                                       - 文章标题（需包含技术点）
                                       - 文章标签（多个用英文逗号隔开）
                                       - 文章简述（不超过 100 字）
                                    4. 使用钉钉机器人工具将文章标题、简述及保存路径作为纯文本消息发送出去；
                                
                                    请根据以上规则自动规划任务流程，并调用相应的工具完成操作。
                                """
                )).build();
        ChatResponse chatResponse = chatModel.call(prompt);
        log.info("结果：{}", JSON.toJSONString(chatResponse.getResult().getOutput().getText()));
    }

    @Test
    public void test_chat_model_stream() {
        Prompt prompt = Prompt.builder()
                .messages(new UserMessage(
                        "有哪些工具可以使用呢？"
                )).build();
        Flux<ChatResponse> chatResponseFlux = chatModel.stream(prompt);
        chatResponseFlux.doOnNext(chatResponse -> log.info("结果：{}", JSON.toJSONString(chatResponse.getResults())))
                .blockLast();
    }

    @Test
    public void test_mcp_sync_client() {
        String userInput = "生成一篇有关Java知识的文章";
        log.info("用户输入：{}", userInput);
        String content = chatClient.prompt(userInput)
                .system(s -> s.param("current_date", LocalDate.now().toString()))
                .call()
                .content();
        log.info("AI助手输出：{}", content);

    }

    @Test
    public void test_client03() {
        ChatClient chatClient01 = ChatClient.builder(chatModel)
                .defaultSystem("""
                        你是一个专业的AI提示词优化专家。请帮我优化以下prompt，并按照以下格式返回：
                        
                        # Role: [角色名称]
                        
                        ## Profile
                        - language: [语言]
                        - description: [详细的角色描述]
                        - background: [角色背景]
                        - personality: [性格特征]
                        - expertise: [专业领域]
                        - target_audience: [目标用户群]
                        
                        ## Skills
                        
                        1. [核心技能类别]
                           - [具体技能]: [简要说明]
                           - [具体技能]: [简要说明]
                           - [具体技能]: [简要说明]
                           - [具体技能]: [简要说明]
                        
                        2. [辅助技能类别]
                           - [具体技能]: [简要说明]
                           - [具体技能]: [简要说明]
                           - [具体技能]: [简要说明]
                           - [具体技能]: [简要说明]
                        
                        ## Rules
                        
                        1. [基本原则]：
                           - [具体规则]: [详细说明]
                           - [具体规则]: [详细说明]
                           - [具体规则]: [详细说明]
                           - [具体规则]: [详细说明]
                        
                        2. [行为准则]：
                           - [具体规则]: [详细说明]
                           - [具体规则]: [详细说明]
                           - [具体规则]: [详细说明]
                           - [具体规则]: [详细说明]
                        
                        3. [限制条件]：
                           - [具体限制]: [详细说明]
                           - [具体限制]: [详细说明]
                           - [具体限制]: [详细说明]
                           - [具体限制]: [详细说明]
                        
                        ## Workflows
                        
                        - 目标: [明确目标]
                        - 步骤 1: [详细说明]
                        - 步骤 2: [详细说明]
                        - 步骤 3: [详细说明]
                        - 预期结果: [说明]
                        
                        
                        ## Initialization
                        作为[角色名称]，你必须遵守上述Rules，按照Workflows执行任务。
                        
                        请基于以上模板，优化并扩展以下prompt，确保内容专业、完整且结构清晰，注意不要携带任何引导词或解释，不要使用代码块包围。
                        """)
                .defaultAdvisors(
                        PromptChatMemoryAdvisor.builder(
                                MessageWindowChatMemory.builder()
                                        .maxMessages(100)
                                        .build()
                        ).build(),
                        // TODO 如果去掉 new RagAnswerAdvisor 这部分就可以正常处理整个流程
                        new RagAnswerAdvisor(pgVectorStore, SearchRequest.builder()
                                .topK(5)
                                .filterExpression("knowledge == 'article-prompt-words'")
                                .build())
                )
                .defaultOptions(OpenAiChatOptions.builder()
                        .model("qwen2.5-7b-instruct-1m")
                        .build())
                .build();

        log.info("\n用户输入：{}", "生成一篇有关Java基础知识的文章");
        String content = chatClient01
                .prompt("生成一篇文章Java基础知识文档")
                .system(s -> s.param("current_date", LocalDate.now().toString()))
                .advisors(a -> a
                        .param(CHAT_MEMORY_CONVERSATION_ID_KEY, "chatId-101")
                        .param(CHAT_MEMORY_RETRIEVE_SIZE_KEY, 100))
                .call().content();

        log.info("\nAI助手输出：{}", content);

        ChatClient chatClient02 = ChatClient.builder(chatModel)
                .defaultSystem("""
                            你是一个 AI Agent 智能体，可以根据用户输入的信息自动生成 Markdown 技术文章，并通过文件系统工具保存，最后使用钉钉机器人工具JavaSDKMCPClient_send_text_message推送通知。今天是 {current_date}。
                        
                            你擅长使用 Planning 模式来分步骤完成任务，具体流程如下：
                        
                            1. 分析用户的输入内容，理解需求并生成结构化的 Markdown 技术文章；
                            2. 使用文件系统工具，保存文件路径为：E:/桌面文件/记事本/mcp_file_system；创建.md文件，并将文章内容写入该文件；
                            3. 提取以下结构化信息：
                               - 文章标题（需包含技术点）
                               - 文章标签（多个用英文逗号隔开）
                               - 文章简述（不超过 100 字）
                            4. 使用钉钉机器人工具将文章标题、简述及保存路径作为纯文本消息发送出去；
                        
                            请根据以上规则自动规划任务流程，并调用相应的工具完成操作。
                        """)
                .defaultAdvisors(
                        PromptChatMemoryAdvisor.builder(
                                MessageWindowChatMemory.builder()
                                        .maxMessages(100)
                                        .build()
                        ).build(),
                        new SimpleLoggerAdvisor()
                )
                .defaultOptions(OpenAiChatOptions.builder()
                        .model("qwen2.5-7b-instruct-1m")
                        .build())
                .build();

        String userInput = "生成一篇文章，要求如下 \r\n" + content;
        log.info("\nUser Input: {}", userInput);
        String assistantResponse = chatClient02
                .prompt(userInput)
                .system(s -> s.param("current_date", LocalDate.now().toString()))
                .advisors(a -> a
                        .param(CHAT_MEMORY_CONVERSATION_ID_KEY, "chatId-101")
                        .param(CHAT_MEMORY_RETRIEVE_SIZE_KEY, 100))
                .call().content();

        log.info("\nAI Assistant Response: {}", assistantResponse);
    }


    public McpSyncClient fileSystemMcpClient() {
        ServerParameters fileSystemMcpParams = ServerParameters.builder("npx.cmd")
                .args("-y", "@modelcontextprotocol/server-filesystem", "E:/桌面文件/记事本/mcp_file_system", "E:/桌面文件/记事本/mcp_file_system")
                .build();
        McpSyncClient mcpClient = McpClient.sync(new StdioClientTransport(fileSystemMcpParams)).requestTimeout(Duration.ofSeconds(10))
                .build();
        McpSchema.InitializeResult initialize = mcpClient.initialize();

        log.info("MCP Server Initialized: {}", initialize);
        return mcpClient;
    }

    public McpSyncClient mcpDingdingBotClient() {
        Map<String, String> envMap = new HashMap<>();
        envMap.put("DINGTALK_BOT_ACCESS_TOKEN", "e041bcba7c912e5bbbb5959d04cf66adb3ca58516d18b0bd5ac26966ba90b436");
        envMap.put("DINGTALK_BOT_SECRET", "SECce8002f3c813906e0202191877141d8931d156fb8349c45ccd4c9f8aa8a379ef");

        ServerParameters dingdingBotMcpParams = ServerParameters.builder("npx.cmd")
                .args("-y", "mcp-dingding-bot")
                .env(envMap)
                .build();
        McpSyncClient mcpClient = McpClient.sync(new StdioClientTransport(dingdingBotMcpParams)).requestTimeout(Duration.ofSeconds(10))
                .build();
        McpSchema.InitializeResult initialize = mcpClient.initialize();
        log.info("MCP Server Initialized: {}", initialize);
        return mcpClient;
    }
}
