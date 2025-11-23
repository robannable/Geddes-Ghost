# Creating a New Repository: "With The Ancients"

This guide explains how to transform this branch into a completely new repository while preserving all the work done.

## Method 1: GitHub Fork and Branch (Recommended)

This approach maintains a connection to the original GeddesGhost repository while creating a new standalone project.

### Steps:

1. **Merge this branch into a new branch on your fork:**
   ```bash
   # Ensure you're on the current branch
   git checkout claude/historical-figure-selector-011VGyUPFxNhVC798vTvBMwH

   # Create and checkout a new clean branch called 'main'
   git checkout -b main

   # Push this new main branch to your fork
   git push -u origin main
   ```

2. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Repository name: `with-the-ancients`
   - Description: "An interactive AI system for conversations with historical figures"
   - Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

3. **Add the new repository as a remote and push:**
   ```bash
   # Add new repository as remote
   git remote add new-origin git@github.com:YOUR_USERNAME/with-the-ancients.git

   # Push all branches to new repository
   git push new-origin main

   # Optional: Push the development branch too
   git push new-origin claude/historical-figure-selector-011VGyUPFxNhVC798vTvBMwH
   ```

4. **Update the remote origin (optional):**
   ```bash
   # If you want this local repo to point to the new repository
   git remote remove origin
   git remote rename new-origin origin

   # Verify
   git remote -v
   ```

5. **Set main as default branch on GitHub:**
   - Go to repository Settings → Branches
   - Change default branch from any dev branch to `main`

---

## Method 2: Fresh Start (Clean Slate)

This creates a completely new repository with no connection to the original.

### Steps:

1. **Create the new repository on GitHub:**
   - Go to https://github.com/new
   - Repository name: `with-the-ancients`
   - Description: "An interactive AI system for conversations with historical figures"
   - Choose Public or Private
   - **DO NOT** initialize with README
   - Click "Create repository"

2. **Export your current work to a new directory:**
   ```bash
   # From the current directory, create a clean export
   cd /tmp
   git clone /home/user/Geddes-Ghost with-the-ancients-clean
   cd with-the-ancients-clean

   # Checkout your development branch
   git checkout claude/historical-figure-selector-011VGyUPFxNhVC798vTvBMwH

   # Create a fresh main branch
   git checkout -b main

   # Remove old remote
   git remote remove origin

   # Add new remote
   git remote add origin git@github.com:YOUR_USERNAME/with-the-ancients.git

   # Push to new repository
   git push -u origin main
   ```

3. **Clean up the history (optional but recommended):**
   ```bash
   # If you want to start with a single clean commit
   git checkout --orphan temp-main
   git add -A
   git commit -m "Initial commit: With The Ancients v2.0

   Multi-character conversation system for dialogue with historical figures.

   Features:
   - Single character, multiple perspectives, and dialogue modes
   - Character configuration system
   - Patrick Geddes and Jane Jacobs included
   - RAG-based knowledge retrieval
   - Admin dashboard for analytics

   Inspired by Helen de Cruz's 'Thinking with the Ancients'
   Original GeddesGhost by Rob Annable (Birmingham School of Architecture)"

   git branch -D main
   git branch -m main
   git push -f origin main
   ```

---

## Method 3: GitHub Import (Simplest)

Use GitHub's import feature to bring in just the code you want.

### Steps:

1. **Download the current branch as a ZIP:**
   ```bash
   # Create a clean working copy
   git archive --format=zip --output=/tmp/with-the-ancients.zip HEAD
   ```

2. **Create new repository on GitHub:**
   - Go to https://github.com/new
   - Create `with-the-ancients`
   - Initialize with README (you'll replace it)

3. **Upload your code:**
   ```bash
   # Clone the new empty repository
   git clone git@github.com:YOUR_USERNAME/with-the-ancients.git
   cd with-the-ancients

   # Extract your work (overwriting the default README)
   unzip /tmp/with-the-ancients.zip

   # Add and commit
   git add -A
   git commit -m "Initial commit: With The Ancients v2.0"
   git push origin main
   ```

---

## Post-Creation Checklist

After creating your new repository, complete these steps:

### 1. Update Repository Settings

- [ ] Add description: "An interactive AI system for conversations with historical figures"
- [ ] Add topics: `ai`, `llm`, `education`, `philosophy`, `historical-figures`, `claude`, `streamlit`, `rag`
- [ ] Add website (if you deploy it)
- [ ] Enable Issues
- [ ] Enable Discussions (recommended for educational use)

### 2. Update Documentation

- [ ] Verify README.md has correct clone URL
- [ ] Update any remaining references from old repo name
- [ ] Add LICENSE file (MIT recommended)
- [ ] Create CONTRIBUTING.md if accepting contributions

### 3. Configure Branch Protection (Optional)

- [ ] Protect `main` branch
- [ ] Require pull request reviews
- [ ] Require status checks

### 4. Set Up GitHub Pages (Optional)

If you want to host documentation:
- [ ] Create `docs/` directory
- [ ] Enable GitHub Pages in settings
- [ ] Point to `docs/` folder

### 5. Add Secrets (for CI/CD)

If you plan to deploy:
- [ ] Add `ANTHROPIC_API_KEY` secret
- [ ] Add any other required secrets

### 6. Create Initial Release

- [ ] Create a release tagged `v2.0.0`
- [ ] Use "With The Ancients - Initial Release" as title
- [ ] Include changelog and features

---

## Maintaining Both Repositories

If you want to keep both the original GeddesGhost and the new With The Ancients:

```bash
# In your local repository
git remote add geddes-ghost git@github.com:robannable/Geddes-Ghost.git
git remote add ancients git@github.com:YOUR_USERNAME/with-the-ancients.git

# Fetch from both
git fetch geddes-ghost
git fetch ancients

# Work on GeddesGhost
git checkout main  # or appropriate branch
git pull geddes-ghost main

# Work on With The Ancients
git checkout with-ancients-main  # create if needed
git pull ancients main

# Cherry-pick changes between them as needed
git cherry-pick <commit-hash>
```

---

## Recommended Workflow: Fork Strategy

**Best approach for most cases:**

1. Keep original GeddesGhost repository for single-character Geddes work
2. Create new With The Ancients repository for multi-character system
3. Both can coexist and share improvements via cherry-picking

This preserves:
- Original GeddesGhost identity and focus
- Clean separation of concerns
- Ability to share bug fixes between projects
- Clear attribution and history

---

## Important Notes

### What Gets Preserved

✅ All commit history (unless you clean it)
✅ All files and directories
✅ Character configurations
✅ Documentation
✅ Scripts and utilities

### What Needs Updating

After creating new repo, update these files with correct URLs:

- `README.md` - Clone URL
- `MULTI_CHARACTER_SYSTEM.md` - Any repo-specific links
- `.github/workflows/*` - If you add CI/CD
- Any documentation with GitHub URLs

### Attribution

Please maintain attribution:
- Original GeddesGhost by Rob Annable
- Inspiration from Helen de Cruz
- Birmingham School of Architecture acknowledgment

This can be in README, LICENSE, or ACKNOWLEDGMENTS file.

---

## Example Complete Workflow

Here's a complete example of Method 1 (Recommended):

```bash
# 1. Ensure you're on the development branch
cd /home/user/Geddes-Ghost
git checkout claude/historical-figure-selector-011VGyUPFxNhVC798vTvBMwH

# 2. Create a clean main branch
git checkout -b main

# 3. Create new GitHub repository (do this on GitHub web interface)
# Name: with-the-ancients
# Don't initialize

# 4. Add new repository as remote
git remote add ancients git@github.com:YOUR_USERNAME/with-the-ancients.git

# 5. Push to new repository
git push -u ancients main

# 6. Update local repo to point to new remote (optional)
git remote rename origin old-origin
git remote rename ancients origin

# 7. Verify
git remote -v
# Should show:
# origin  git@github.com:YOUR_USERNAME/with-the-ancients.git (fetch)
# origin  git@github.com:YOUR_USERNAME/with-the-ancients.git (push)

# 8. Update README clone URL
sed -i 's|yourusername/geddes-ghost|YOUR_USERNAME/with-the-ancients|g' README.md
git add README.md
git commit -m "Update repository URLs"
git push origin main

# Done! Your new repository is ready.
```

---

## Troubleshooting

### "Repository not found" error
- Check remote URL: `git remote -v`
- Verify repository exists on GitHub
- Check SSH keys: `ssh -T git@github.com`

### "Permission denied" error
- Use HTTPS instead: `https://github.com/USERNAME/with-the-ancients.git`
- Or set up SSH keys

### "Branch doesn't exist" error
- Create it first: `git checkout -b main`
- Push it: `git push -u origin main`

### Large files warning
- Check .gitignore excludes logs/, .venv/
- Use git-lfs for large PDFs if needed

---

## Next Steps After Repository Creation

1. **Test the installation** from scratch:
   ```bash
   git clone git@github.com:YOUR_USERNAME/with-the-ancients.git
   cd with-the-ancients
   ./run_ancients.sh
   ```

2. **Create a Release**:
   - Tag: `v2.0.0`
   - Title: "With The Ancients - Multi-Character System"
   - Description: Feature list and acknowledgments

3. **Share**:
   - Tweet/post about it with #WithTheAncients
   - Share in academic circles
   - Add to awesome-lists if applicable

4. **Deploy** (optional):
   - Streamlit Cloud
   - Heroku
   - Railway
   - Your own server

---

**Congratulations!** You now have a clean, standalone "With The Ancients" repository! 🎉
