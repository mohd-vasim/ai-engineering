## Full Steps: `git init` in Child → Convert to Submodule

---

### Step 1: Go into child folder & initialize git

```bash
cd /Users/vasim/Programming/ai-engineering/shoplifting-detection

git init
git add .
git commit -m "Initial commit"
```

---

### Step 2: Add your new GitHub remote & push

```bash
git remote add origin https://github.com/mohd-vasim/shoplifting-detection.git

git branch -M main
git push -u origin main
```

---

### Step 3: Go to parent & remove child from its tracking

```bash
cd /Users/vasim/Programming/ai-engineering

# This untracks the folder but keeps files on disk
git rm -r --cached shoplifting-detection/

git commit -m "Remove shoplifting-detection before adding as submodule"
```

---

### Step 4: Register child as submodule in parent

```bash
# Still in parent folder
git submodule add https://github.com/mohd-vasim/shoplifting-detection.git shoplifting-detection

git commit -m "Add shoplifting-detection as submodule"
git push origin main
```

---

### Verify it worked

```bash
cat .gitmodules
```

Should output:
```ini
[submodule "shoplifting-detection"]
    path = shoplifting-detection
    url = https://github.com/mohd-vasim/shoplifting-detection.git
```

---

### After setup — daily workflow

```bash
# To push changes in shoplifting-detection
cd shoplifting-detection
git add .
git commit -m "your message"
git push                      # goes to mohd-vasim/shoplifting-detection

# To update the parent's submodule pointer
cd ..
git add shoplifting-detection
git commit -m "Update submodule pointer"
git push                      # goes to mohd-vasim/ai-engineering
```

> 💡 **Each folder now has its own independent git history and remote URL.** They are completely separate repos linked together via the submodule reference.