package com.finrag.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Map;

/**
 * RagClient — makes HTTP calls to the Python FastAPI service.
 *
 * This is the bridge between the Java world and the Python RAG world.
 * The Java controller calls methods on this class; this class handles
 * the HTTP request/response with the Python service.
 *
 * We use RestTemplate (Spring's built-in HTTP client) — simple and
 * sufficient for v1. In a high-throughput production system you'd
 * switch to WebClient (async/reactive) but RestTemplate is easier
 * to understand as a first project.
 *
 * Why a separate Service class and not call HTTP directly from the controller?
 *   Controllers should only handle HTTP concerns (request/response mapping).
 *   Business logic (including external calls) belongs in a Service.
 *   This separation makes testing and future changes much easier.
 */
@Slf4j
@Service
public class RagClient {

    private final RestTemplate restTemplate;
    private final String       ragServiceUrl;

    public RagClient(
        @Value("${finrag.rag-service-url}") String ragServiceUrl
    ) {
        this.restTemplate  = new RestTemplate();
        this.ragServiceUrl = ragServiceUrl;
    }

    /**
     * Call POST /query on the Python service.
     *
     * Sends the user's question and list of tickers.
     * Returns the full response body as a Map (JSON → Java Map).
     *
     * The Map contains: answer, sources, companies, chunks_used, model
     */
    public Map<?, ?> query(String userQuery, List<String> tickers) {
        String url = ragServiceUrl + "/query";

        // Build the JSON request body
        Map<String, Object> body = Map.of(
            "query",   userQuery,
            "tickers", tickers != null ? tickers : List.of()
        );

        log.debug("Calling RAG /query: tickers={} query='{}'",
                  tickers, userQuery.substring(0, Math.min(80, userQuery.length())));

        return post(url, body);
    }

    /**
     * Call POST /ingest on the Python service.
     *
     * Triggers ingestion for one company (10-K + news).
     * Returns: {"ticker": "AAPL", "chunks_stored": 412, "status": "ok"}
     *
     * This is a slow operation (1-5 min) — RestTemplate will wait.
     * In production, convert this to an async job with a status endpoint.
     */
    public Map<?, ?> ingest(String ticker, String companyName) {
        String url = ragServiceUrl + "/ingest";

        Map<String, Object> body = Map.of(
            "ticker",       ticker.toUpperCase(),
            "company_name", companyName
        );

        log.info("Triggering ingestion for ticker={} company='{}'",
                 ticker, companyName);

        return post(url, body);
    }

    /**
     * Shared POST helper — sets JSON headers and handles errors.
     */
    private Map<?, ?> post(String url, Map<String, Object> body) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);

        try {
            ResponseEntity<Map> response = restTemplate.postForEntity(
                url, request, Map.class
            );
            return response.getBody();

        } catch (HttpClientErrorException e) {
            // 4xx from Python service — pass the error message through
            log.error("RAG service returned {}: {}", e.getStatusCode(), e.getResponseBodyAsString());
            throw new RagServiceException(
                "RAG service error: " + e.getStatusCode() + " — " + e.getResponseBodyAsString(),
                e.getStatusCode().value()
            );
        } catch (Exception e) {
            // Network error, timeout, Python service down, etc.
            log.error("Failed to reach RAG service at {}: {}", url, e.getMessage());
            throw new RagServiceException(
                "Could not reach the RAG service. Is it running at " + ragServiceUrl + "?",
                503
            );
        }
    }
}
