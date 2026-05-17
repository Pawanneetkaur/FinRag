package com.finrag.controller;

import com.finrag.service.RagClient;
import com.finrag.service.RagServiceException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * QueryController — handles financial Q&A requests from users.
 *
 * Endpoint: POST /api/query
 *
 * This controller is intentionally thin — it validates the request,
 * delegates to RagClient, and maps errors to HTTP responses.
 * All the AI logic is in the Python service.
 *
 * @RestController = @Controller + @ResponseBody (returns JSON automatically)
 * @RequestMapping sets the base path for all methods in this class
 */
@Slf4j
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class QueryController {

    private final RagClient ragClient;

    /**
     * Ask a financial question about one or more companies.
     *
     * Request body:
     *   {
     *     "query":   "Compare Apple and Microsoft revenue growth in 2023",
     *     "tickers": ["AAPL", "MSFT"]
     *   }
     *
     * Response body:
     *   {
     *     "answer":      "## AAPL\nApple's revenue grew...",
     *     "sources":     [...],
     *     "companies":   ["AAPL", "MSFT"],
     *     "chunks_used": 6,
     *     "model":       "claude-sonnet-4-20250514"
     *   }
     *
     * The JWT in the Authorization header is already validated by AuthFilter
     * before this method is called — no auth code needed here.
     */
    @PostMapping("/query")
    public ResponseEntity<?> query(@RequestBody QueryRequest req) {

        if (req.query() == null || req.query().isBlank()) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "Query cannot be empty."));
        }

        log.info("Query request: tickers={} query='{}'",
                 req.tickers(), req.query().substring(0, Math.min(80, req.query().length())));

        try {
            Map<?, ?> result = ragClient.query(req.query(), req.tickers());
            return ResponseEntity.ok(result);

        } catch (RagServiceException e) {
            return ResponseEntity.status(e.getStatusCode())
                .body(Map.of("error", e.getMessage()));
        }
    }

    // Java record — a concise immutable data class (Java 16+)
    // Replaces a full POJO with getters, equals, hashCode automatically
    record QueryRequest(String query, List<String> tickers) {}
}
