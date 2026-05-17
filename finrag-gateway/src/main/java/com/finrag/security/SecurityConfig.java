package com.finrag.security;

import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

/**
 * SecurityConfig — defines which endpoints require auth and registers our JWT filter.
 *
 * Spring Security by default protects everything and uses sessions.
 * We override both: JWTs are stateless (no sessions needed).
 *
 * Public endpoints:
 *   GET  /health         — liveness probe (no auth needed)
 *   POST /auth/login     — get a JWT token (no auth needed — obviously)
 *
 * Protected endpoints:
 *   POST /api/query      — requires valid JWT
 *   POST /api/ingest     — requires valid JWT
 */
@Configuration
@EnableWebSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    private final AuthFilter authFilter;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            // Disable CSRF — not needed for stateless JWT APIs
            // CSRF protects session-based apps; JWTs don't use sessions
            .csrf(csrf -> csrf.disable())

            // Stateless session — Spring Security won't create HTTP sessions
            // Every request must carry its own JWT
            .sessionManagement(session ->
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            )

            // Define which endpoints are public vs protected
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/health", "/auth/login").permitAll()
                .anyRequest().authenticated()
            )

            // Register our JWT filter BEFORE Spring's default auth filter.
            // This means JWT validation happens first — if valid, the request
            // is already authenticated by the time Spring's filter runs.
            .addFilterBefore(authFilter, UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }
}
