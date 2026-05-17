package com.finrag;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * FinRAG Gateway — Spring Boot entry point.
 *
 * @SpringBootApplication is a shortcut for three annotations:
 *   @Configuration      — this class can define Spring beans
 *   @EnableAutoConfig   — Spring auto-configures based on classpath
 *   @ComponentScan      — scans com.finrag.* for controllers, services, etc.
 *
 * Running this class starts the embedded Tomcat server on port 8080.
 * The Python RAG service runs separately on port 8000.
 */
@SpringBootApplication
public class FinragApplication {

    public static void main(String[] args) {
        SpringApplication.run(FinragApplication.class, args);
    }
}
