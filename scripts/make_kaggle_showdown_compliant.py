import re

def process_content_for_showdown(content, is_kaggle_export=False):
    # 1. If is_kaggle_export, replace relative image links with raw.githubusercontent.com URLs
    if is_kaggle_export:
        github_base = "https://raw.githubusercontent.com/xstarbrain/dual-loop-controller/main"
        content = content.replace(
            "![X-Star HADL v5.9 Card Thumbnail](card_thumbnail.png)",
            f"![X-Star HADL v5.9 Card Thumbnail]({github_base}/card_thumbnail.png)"
        )
        content = content.replace(
            "![100Q Novel Challenge Comparison](figures/benchmark_100_novel_challenges_comparison.png)",
            f"![100Q Novel Challenge Comparison]({github_base}/kaggle_writeup/figures/benchmark_100_novel_challenges_comparison.png)"
        )
        content = content.replace(
            "![400Q Canonical Suite](figures/benchmark_400q_hadl_v59_comparison.png)",
            f"![400Q Canonical Suite]({github_base}/kaggle_writeup/figures/benchmark_400q_hadl_v59_comparison.png)"
        )
        content = content.replace(
            "![Architecture Blueprint](figures/hadl_v58_architecture_blueprint.png)",
            f"![Architecture Blueprint]({github_base}/kaggle_writeup/figures/hadl_v58_architecture_blueprint.png)"
        )
        # Also badge audit links
        content = content.replace(
            "data/benchmark_100_novel_challenges_audit.json",
            "https://github.com/xstarbrain/dual-loop-controller/blob/main/kaggle_writeup/data/benchmark_100_novel_challenges_audit.json"
        )
        content = content.replace(
            "data/benchmark_400q_hadl_v59_audit.json",
            "https://github.com/xstarbrain/dual-loop-controller/blob/main/kaggle_writeup/data/benchmark_400q_hadl_v59_audit.json"
        )
        content = content.replace(
            "../LICENSE",
            "https://github.com/xstarbrain/dual-loop-controller/blob/main/LICENSE"
        )

    # 2. Escape underscores inside display math blocks $$...$$
    def escape_underscores_in_display(match):
        block = match.group(1)
        # Escape any unescaped underscore: (?<!\\)_ -> \_
        escaped_block = re.sub(r"(?<!\\)_", r"\\_", block)
        return f"\n\n$$\n{escaped_block.strip()}\n$$\n\n"

    # Match $$...$$ blocks
    content = re.sub(r"\$\$\s*([\s\S]*?)\s*\$\$", escape_underscores_in_display, content)

    # 3. Escape underscores inside inline math $...$ (single dollar)
    # Be careful not to match across lines or $$
    def escape_underscores_in_inline(match):
        inline = match.group(1)
        if "\n" in inline:
            return match.group(0)
        escaped_inline = re.sub(r"(?<!\\)_", r"\\_", inline)
        return f"${escaped_inline}$"

    # Match single-dollar inline math: (?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)
    content = re.sub(r"(?<![\$\\])\$(?!\$)([^\$\n]+?)(?<!\\)\$", escape_underscores_in_inline, content)

    # 4. Clean up any excess whitespace
    content = re.sub(r"\n{4,}", "\n\n", content)

    return content

def run():
    # Process kaggle_writeup/writeup.md as Kaggle export
    with open("kaggle_writeup/writeup.md", "r", encoding="utf-8") as f:
        kw = f.read()
    kw_fixed = process_content_for_showdown(kw, is_kaggle_export=True)
    with open("kaggle_writeup/writeup.md", "w", encoding="utf-8") as f:
        f.write(kw_fixed)
    print("Updated kaggle_writeup/writeup.md with Showdown escaping & absolute GitHub image URLs!")

    # Process WRITEUP.md (root) with Showdown escaping & relative paths for GitHub
    with open("WRITEUP.md", "r", encoding="utf-8") as f:
        w = f.read()
    w_fixed = process_content_for_showdown(w, is_kaggle_export=False)
    with open("WRITEUP.md", "w", encoding="utf-8") as f:
        f.write(w_fixed)
    print("Updated WRITEUP.md with Showdown escaping!")

    # Process kaggle_writeup/README.md
    with open("kaggle_writeup/README.md", "r", encoding="utf-8") as f:
        r = f.read()
    r_fixed = process_content_for_showdown(r, is_kaggle_export=False)
    with open("kaggle_writeup/README.md", "w", encoding="utf-8") as f:
        f.write(r_fixed)
    print("Updated kaggle_writeup/README.md with Showdown escaping!")

if __name__ == "__main__":
    run()
