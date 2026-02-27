## Analysis

**Root Cause:** The repository lacks an `.editorconfig` file, which means developers' editors/IDEs use their personal defaults for indentation, line endings, and encoding. This leads to:
- Inconsistent code formatting across files
- Noisy diffs (whitespace changes mixed with logic changes)
- Cross-platform issues (Windows CRLF vs Unix LF)

## Solution

Create an `.editorconfig` file at the repository root with sensible defaults:

```ini
# EditorConfig is awesome: https://EditorConfig.org

# top-most EditorConfig file
root = true

# Default settings for all files
[*]
charset = utf-8
end_of_line = lf
indent_size = 2
indent_style = space
insert_final_newline = true
trim_trailing_whitespace = true

# Markdown files preserve trailing whitespace (used for line breaks)
[*.md]
trim_trailing_whitespace = false

# Makefiles require tabs
[{Makefile,*.mk}]
indent_style = tab
```

## Explanation

- **`root = true`**: Stops EditorConfig from searching parent directories for other config files
- **`charset = utf-8`**: Prevents encoding issues with special characters
- **`end_of_line = lf`**: Ensures Unix-style line endings (critical for Docker/scripts and cross-platform teams)
- **`indent_size = 2` / `indent_style = space`**: Modern standard for web/JS projects; change to `4` if this is a Python/Java backend project
- **`insert_final_newline = true`**: POSIX standard requirement; prevents "No newline at end of file" warnings
- **`trim_trailing_whitespace = true`**: Keeps diffs clean by removing irrelevant whitespace changes
- **Markdown exception**: Preserves intentional trailing spaces used for line breaks in Markdown
- **Makefile exception**: Make requires actual tab characters, not spaces

## Next Steps

1. Save the file as `.editorconfig` in the repository root
2. Commit the file: `git add .editorconfig && git commit -m "chore: add editorconfig file (#131)"`
3. Most modern editors (VS Code, IntelliJ, Vim with plugin) will automatically respect these settings

*Note: Adjust `indent_size` to `4` if this is primarily a Python, Java, or C# repository.*
