package proxies

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"strings"
)

func RunReadOnlyProxy() error {
	listenSocket := getEnv("LISTEN_SOCKET", defaultROListenSocket)
	upstreamSocket := getEnv("UPSTREAM_SOCKET", defaultUpstreamSocket)

	if err := ensureParentDir(listenSocket); err != nil {
		return fmt.Errorf("failed to prepare directory for %s: %w", listenSocket, err)
	}
	removeIfExists(listenSocket)

	proxy := newUnixReverseProxy(upstreamSocket, allowReadOnly)

	listener, err := net.Listen("unix", listenSocket)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", listenSocket, err)
	}
	ensureSocketMode(listenSocket, socketPermissions)

	server := &http.Server{Handler: logRequests(proxy)}

	log.Printf("Read-only proxy listening on %s -> %s", listenSocket, upstreamSocket)
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("proxy server error: %w", err)
	}
	return nil
}

func allowReadOnly(r *http.Request, body []byte) error {
	switch r.Method {
	case http.MethodGet, http.MethodHead, http.MethodOptions:
		return nil
	case http.MethodPost:
		if isCursorPath(r.URL.Path) {
			upper := strings.ToUpper(string(body))
			forbiddenKeywords := []string{"INSERT", "UPDATE", "UPSERT", "REMOVE", "REPLACE", "TRUNCATE", "DROP"}
			for _, keyword := range forbiddenKeywords {
				if strings.Contains(upper, keyword) {
					return fmt.Errorf("forbidden keyword %q detected in request body", keyword)
				}
			}
			return nil
		}
	case http.MethodPut, http.MethodDelete:
		if isCursorPath(r.URL.Path) {
			return nil
		}
	}
	return fmt.Errorf("method %s not permitted on %s", r.Method, r.URL.Path)
}

func isCursorPath(path string) bool {
	// Match .../_api/cursor
	return strings.Contains(path, "/_api/cursor")
}
