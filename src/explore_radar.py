import sqlite3
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def main():
    db = sqlite3.connect("state/notebook_radar/radar.sqlite3")
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT ref, title, author, category, status, best_public_score, public_score, public_votes, tags_json
        FROM notebooks
        ORDER BY best_public_score DESC NULLS LAST, public_votes DESC
        LIMIT 40
    """).fetchall()

    print(f"Total top notebooks fetched: {len(rows)}")
    for r in rows:
        title = r["title"] or ""
        author = r["author"] or ""
        ref = r["ref"] or ""
        score = r["best_public_score"]
        cat = r["category"] or ""
        status = r["status"] or ""
        votes = r["public_votes"] or 0
        print(f"[{score}] ({status}/{cat}) {ref} | votes:{votes} | {title}")

if __name__ == "__main__":
    main()
