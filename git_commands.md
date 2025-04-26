## Overview

The command below generates a concise summary of every contributor’s commit count in the current Git repository:

```bash
git shortlog -s -n
```

---

## 2. List every commit (with messages) by a given author

You can view all commits made by a specific author, including the commit hash, date, and message, using the following command:

```bash
git log --author="USERNAME" --pretty=format:"%h %ad %s" --date=short
```

**Explanation:**
- **USERNAME**: The author’s name or email (case-insensitive)
- **%h**: Abbreviated commit hash
- **%ad**: Commit date (formatted as YYYY-MM-DD)
- **%s**: Commit message subject
```

---

✅ Now if you paste this into your markdown editor, it will render perfectly — just like you're expecting!  
Would you also want me to give a version that includes a sample output so it looks even more polished? 📄✨