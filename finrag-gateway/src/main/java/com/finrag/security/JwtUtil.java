package com.finrag.security;

import io.jsonwebtoken.*;
import io.jsonwebtoken.security.Keys;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.Date;

/**
 * JwtUtil — creates and validates JWT tokens.
 *
 * A JWT has three parts separated by dots:
 *   header.payload.signature
 *
 * Header:    algorithm used (HS256)
 * Payload:   claims — who the user is, when the token expires
 * Signature: HMAC of header+payload using our secret key
 *
 * Validation: we recompute the signature and check it matches.
 * If anyone tampers with the payload, the signature won't match → rejected.
 *
 * We never store JWTs — users present them on every request.
 */
@Component
public class JwtUtil {

    private final SecretKey key;
    private final long      expirationMs;

    public JwtUtil(
        @Value("${finrag.jwt-secret}")         String secret,
        @Value("${finrag.jwt-expiration-ms}")  long   expirationMs
    ) {
        // Keys.hmacShaKeyFor requires at least 256-bit key for HS256
        this.key          = Keys.hmacShaKeyFor(secret.getBytes(StandardCharsets.UTF_8));
        this.expirationMs = expirationMs;
    }

    /**
     * Generate a signed JWT for a given username.
     * Called by AuthController at login time.
     *
     * Claims stored in the token:
     *   sub  — subject (username)
     *   iat  — issued at timestamp
     *   exp  — expiry timestamp
     */
    public String generateToken(String username) {
        Date now    = new Date();
        Date expiry = new Date(now.getTime() + expirationMs);

        return Jwts.builder()
            .subject(username)
            .issuedAt(now)
            .expiration(expiry)
            .signWith(key)
            .compact();
    }

    /**
     * Extract the username (subject) from a token.
     * Used by the JWT filter to identify the user.
     */
    public String extractUsername(String token) {
        return parseClaims(token).getPayload().getSubject();
    }

    /**
     * Validate a token — returns true if the signature is valid
     * and the token is not expired.
     */
    public boolean isValid(String token) {
        try {
            parseClaims(token);
            return true;
        } catch (JwtException | IllegalArgumentException e) {
            // Covers: expired, malformed, wrong signature, null
            return false;
        }
    }

    private Jws<Claims> parseClaims(String token) {
        return Jwts.parser()
            .verifyWith(key)
            .build()
            .parseSignedClaims(token);
    }
}
