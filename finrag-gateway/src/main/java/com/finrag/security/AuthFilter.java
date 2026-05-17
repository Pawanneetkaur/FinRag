package com.finrag.security;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.List;

/**
 * AuthFilter — runs once per HTTP request before the controller is called.
 *
 * What it does:
 *   1. Reads the Authorization header: "Bearer <token>"
 *   2. Validates the JWT signature and expiry
 *   3. If valid, puts the user's identity into the SecurityContext
 *      (Spring Security then allows the request through)
 *   4. If invalid or missing, does nothing (Spring Security will reject it)
 *
 * OncePerRequestFilter guarantees this filter runs exactly once per request,
 * even in async or forwarded requests.
 *
 * @Slf4j       — injects a 'log' logger (Lombok)
 * @Component   — Spring picks this up automatically as a bean
 * @RequiredArgsConstructor — Lombok generates constructor for final fields
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class AuthFilter extends OncePerRequestFilter {

    private final JwtUtil jwtUtil;

    @Override
    protected void doFilterInternal(
        HttpServletRequest  request,
        HttpServletResponse response,
        FilterChain         chain
    ) throws ServletException, IOException {

        String authHeader = request.getHeader("Authorization");

        // If no Bearer token is present, skip JWT logic entirely.
        // Spring Security's config will reject unauthenticated requests
        // to protected endpoints (everything except /auth/login and /health).
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            chain.doFilter(request, response);
            return;
        }

        String token = authHeader.substring(7);   // strip "Bearer " prefix

        if (!jwtUtil.isValid(token)) {
            log.warn("Invalid or expired JWT token from {}", request.getRemoteAddr());
            chain.doFilter(request, response);
            return;
        }

        String username = jwtUtil.extractUsername(token);
        log.debug("Authenticated user: {}", username);

        // Register the user in the SecurityContext.
        // Spring Security uses this to decide whether to allow the request.
        // We pass empty authorities — role-based access can be added later.
        var auth = new UsernamePasswordAuthenticationToken(
            username,
            null,
            List.of()    // authorities / roles — extend this for RBAC later
        );
        SecurityContextHolder.getContext().setAuthentication(auth);

        chain.doFilter(request, response);
    }
}
