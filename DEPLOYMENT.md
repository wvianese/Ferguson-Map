# How to Deploy Ferguson Housing Map Online

## Option 1: GitHub Pages (Recommended - Free & Easy)

### Steps:

1. **Create a GitHub repository** (if not already online):
   ```bash
   # In your terminal, from this folder:
   git add index.html ferguson_map.html
   git commit -m "Add interactive map for deployment"
   git push origin main
   ```

2. **Enable GitHub Pages**:
   - Go to your repository on GitHub.com
   - Click **Settings** → **Pages** (in left sidebar)
   - Under "Source", select **main** branch
   - Click **Save**

3. **Access your map**:
   - Your map will be live at: `https://YOUR-USERNAME.github.io/REPO-NAME/`
   - Takes 2-3 minutes to deploy first time

### Important Notes:
- The `index.html` is your landing page
- The `ferguson_map.html` is the actual interactive map
- Both files need to be in the repository root
- File is ~200MB but GitHub Pages can handle it

---

## Option 2: Netlify (Alternative - Also Free)

1. Go to [netlify.com](https://netlify.com) and sign up
2. Drag and drop this folder to deploy
3. Get instant URL like `https://your-site.netlify.app`

---

## Option 3: Vercel (Alternative - Also Free)

1. Go to [vercel.com](https://vercel.com) and sign up
2. Click "New Project" → Import your GitHub repo
3. Deploy with one click
4. Get URL like `https://your-site.vercel.app`

---

## What Gets Deployed:

- `index.html` - Landing page with link to map
- `ferguson_map.html` - Full interactive map (200MB with all data embedded)

The map is completely self-contained - no external dependencies needed!
