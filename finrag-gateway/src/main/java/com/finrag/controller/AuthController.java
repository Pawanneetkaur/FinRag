package com.finrag.controller;

import com.finrag.security.JwtUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * AuthController — issues JWT tokens at login.
 *
 * Endpoint: POST /auth/login
 *
 * For v1 we use hardcoded credentials (username/password in config).
 * In production you'd check against a user database with hashed passwords.
 *
 * Flow:
 *   1. User sends username + password
 *   2. We verify credentials
 *   3. If valid, return a signed JWT
 *   4. User includes JWT in all future requests as "Authorization: Bearer <token>"
 *
 * This endpoint is public (no auth required) — configured in SecurityConfig.
 */
@Slf4j
@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
public class AuthController {

    private final JwtUtil jwtUtil;

    // Hardcoded for v1 — replace with a UserService + database in production
    private static final String DEMO_USERNAME = "finrag-user";
    private static final String DEMO_PASSWORD = "finrag-demo-2024";

    /**
     * Login and receive a JWT token.
     *
     * Request body:
     *   {"username": "finrag-user", "password": "finrag-demo-2024"}
     *
     * Success response:
     *   {"token": "eyJhbGci...", "expiresIn": 86400000}
     *
     * Failure response (401):
     *   {"error": "Invalid credentials"}
     */
    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest req) {

        if (!DEMO_USERNAME.equals(req.username()) ||
            !DEMO_PASSWORD.equals(req.password())) {

            log.warn("Failed login attempt for username='{}'", req.username());
            return ResponseEntity.status(401)
                .body(Map.of("error", "Invalid credentials"));
        }

        String token = jwtUtil.generateToken(req.username());
        log.info("JWT issued for user='{}'", req.username());

        return ResponseEntity.ok(Map.of(
            "token",     token,
            "tokenType", "Bearer"
        ));
    }

    record LoginRequest(String username, String password) {}
}
