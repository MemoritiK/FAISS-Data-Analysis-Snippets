import json
from pathlib import Path
from collections import defaultdict
import difflib
import re

from reform_cat import CORE_CATEGORIES 


class CodeDiscovery:
    def __init__(self, snippets_file: str):
        self.snippets_file = Path(snippets_file)
        self.snippets = self._load_snippets()
        self.category_map = self._map_to_core_categories()

    def _load_snippets(self):
        snippets = []
        with open(self.snippets_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    snippets.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return snippets

    def _map_to_core_categories(self):
        category_map = defaultdict(list)
        for snippet in self.snippets:
            cat_found = False
            snippet_cat = snippet.get("category", "").upper()
            snippet_cat = re.sub('[- _]','',snippet_cat)
            for core_cat, subcats in CORE_CATEGORIES.items():
                if snippet_cat in subcats:
                    category_map[core_cat].append(snippet)
                    cat_found = True
                    break
            if not cat_found:
                category_map["OTHER"].append(snippet)
        return dict(category_map)

    def list_categories(self):
        return list(self.category_map.keys())

    def list_snippets(self, category=None, difficulty=None):
        """
        Return snippets filtered by category and/or difficulty
        """
        results = []
        categories = [category] if category else self.category_map.keys()
        for cat in categories:
            for snippet in self.category_map.get(cat, []):
                if difficulty and snippet.get("difficulty") != difficulty:
                    continue
                results.append(snippet)
        return results

    def search_snippets(self, query: str, category=None, difficulty=None, max_results=10, fuzzy=True):
        """
        Simple fuzzy search on question or tags.
        """
        candidates = self.list_snippets(category=category, difficulty=difficulty)
        if fuzzy:
            # Build mapping of snippet idx to searchable text
            texts = [f"{s.get('question','')} {' '.join(s.get('tags',[]))}" for s in candidates]
            matches = difflib.get_close_matches(query, texts, n=max_results, cutoff=0.4)
            results = []
            for m in matches:
                idx = texts.index(m)
                results.append(candidates[idx])
            return results
        else:
            # Exact substring match
            results = []
            for s in candidates:
                text = f"{s.get('question','')} {' '.join(s.get('tags',[]))}".lower()
                if query.lower() in text:
                    results.append(s)
                    if len(results) >= max_results:
                        break
            return results

SNIPPETS_FILE = Path("..") / "cleaned_snippets.jsonl"

if __name__ == "__main__":
    discovery = CodeDiscovery(f"{SNIPPETS_FILE}")

    print("Available categories:", discovery.list_categories())

    snippets_viz = discovery.list_snippets(category="VISUALIZATION_BASICS")
    print(f"\nFound {len(snippets_viz)} visualization snippets")

    # Example: fuzzy search
    query = "pandas drop missing values"
    results = discovery.search_snippets(query, max_results=5)
    print(f"\nSearch results for '{query}':")
    for r in results:
        print(f"- {r.get('question')} [{r.get('category')}]")
