package com.finrag.service;

import lombok.Getter;

/**
 * RagServiceException — thrown when the Python RAG service returns an error
 * or cannot be reached.
 *
 * Carries an HTTP status code so the controller can forward the right
 * status to the client (e.g. 404 if ticker not found, 503 if service is down).
 */
@Getter
public class RagServiceException extends RuntimeException {

    private final int statusCode;

    public RagServiceException(String message, int statusCode) {
        super(message);
        this.statusCode = statusCode;
    }
}
