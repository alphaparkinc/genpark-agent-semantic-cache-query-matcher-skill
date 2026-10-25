import sys
import json
import re
import math
from collections import Counter

class SemanticCache:
    def __init__(self, similarity_threshold=0.75):
        self.cache = {}
        self.similarity_threshold = similarity_threshold
        self.total_queries = 0
        self.hits = 0

    def _tokenize(self, text):
        return Counter(re.findall(r"\w+", text.lower()))

    def _cosine_similarity(self, c1, c2):
        intersection = set(c1.keys()) & set(c2.keys())
        numerator = sum([c1[x] * c2[x] for x in intersection])
        sum1 = sum([c1[x]**2 for x in c1.keys()])
        sum2 = sum([c2[x]**2 for x in c2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        return float(numerator) / denominator

    def set(self, query, response):
        self.cache[query] = {
            "tokens": self._tokenize(query),
            "response": response,
            "hits": 0
        }
        return True

    def get(self, query):
        self.total_queries += 1
        if query in self.cache:
            self.hits += 1
            self.cache[query]["hits"] += 1
            return {"hit": True, "score": 1.0, "response": self.cache[query]["response"], "exact": True}

        q_tokens = self._tokenize(query)
        best_score = 0.0
        best_match = None

        for key, entry in self.cache.items():
            sim = self._cosine_similarity(q_tokens, entry["tokens"])
            if sim > best_score:
                best_score = sim
                best_match = entry

        if best_match and best_score >= self.similarity_threshold:
            self.hits += 1
            best_match["hits"] += 1
            return {"hit": True, "score": round(best_score, 4), "response": best_match["response"], "exact": False}

        return {"hit": False, "score": round(best_score, 4), "response": None, "exact": False}

    def stats(self):
        return {
            "cached_entries": len(self.cache),
            "total_queries": self.total_queries,
            "cache_hits": self.hits,
            "hit_rate": round(self.hits / self.total_queries, 4) if self.total_queries > 0 else 0.0
        }

engine = SemanticCache()

def handle_rpc(request):
    method = request.get("method")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "cache_set",
                        "description": "Store a query and its LLM response into semantic cache.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "response": {"type": "string"}
                            },
                            "required": ["query", "response"]
                        }
                    },
                    {
                        "name": "cache_get",
                        "description": "Retrieve cached response using semantic similarity matching.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"}
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "cache_stats",
                        "description": "Get hit rate and entry statistics for semantic cache.",
                        "inputSchema": {"type": "object", "properties": {}}
                    }
                ]
            }
        }

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "cache_set":
            res = engine.set(args.get("query", ""), args.get("response", ""))
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps({"status": "stored"})}]}}
        elif name == "cache_get":
            res = engine.get(args.get("query", ""))
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res)}]}}
        elif name == "cache_stats":
            res = engine.stats()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res)}]}}

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_rpc(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}}) + "\n")
            sys.stdout.flush()
