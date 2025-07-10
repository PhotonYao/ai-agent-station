package top.kangyaocoding.ai.config;

import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 描述: 配置TokenTextSplitter
 *
 * @author K·Herbert herbert501@qq.com
 * @since 2025-07-09 14:18
 */
@Configuration
public class TokenTextSplitterConfig {
    @Bean
    public TokenTextSplitter tokenTextSplitter() {
        return new TokenTextSplitter();
    }
}
