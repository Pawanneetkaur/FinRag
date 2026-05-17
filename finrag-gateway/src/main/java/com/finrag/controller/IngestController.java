package com.finrag.controller;

import com.finrag.service.RagClient;
import com.finrag.service.RagServiceException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * IngestController — triggers document ingestion for a company.
 *
 * Endpoint: POST /api/ingest
 *
 * This would typically be called by an admin or a scheduled job,
 * not by end users. In v1 it's open to any authenticated user —
 * add role-based access control (RBAC) later if needed.
 *
 * Note on latency: ingestion takes 1-5 minutes. This endpoint blocks
 * until it's done. In production, you'd return a job ID immediately
 * and let the client poll GET /api/ingest/{jobId}/status.
 * That's a good v2 improvement.
 */
@Slf4j
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class IngestController {

    private final RagClient ragClient;

    /**
     * Trigger ingestion for one company.
     *
     * Request body:
     *   {
     *     "ticker":       "AAPL",
     *     "company_name": "Apple"
     *   }
     *
     * Response body:
     *   {
     *     "ticker":        "AAPL",
     *     "company":       "Apple",
     *     "chunks_stored": 412,
     *     "status":        "ok"
     *   }
     */
    @PostMapping("/ingest")
    public ResponseEntity<?> ingest(@RequestBody IngestRequest req) {

        if (req.ticker() == null || req.ticker().isBlank()) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "Ticker is required."));
        }
        if (req.companyName() == null || req.companyName().isBlank()) {
            return ResponseEntity.badRequest()
                .body(Map.of("error", "Company name is required."));
        }

        log.info("Ingest request: ticker={} company='{}'",
                 req.ticker(), req.companyName());

        try {
            Map<?, ?> result = ragClient.ingest(req.ticker(), req.companyName());
            return ResponseEntity.ok(result);

        } catch (RagServiceException e) {
            return ResponseEntity.status(e.getStatusCode())
                .body(Map.of("error", e.getMessage()));
        }
    }

    record IngestRequest(String ticker, String companyName) {}
}
