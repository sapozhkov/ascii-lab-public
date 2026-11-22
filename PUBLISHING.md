# Publishing guide

This repository shares the source code behind ASCII Lab reels and provides public links for Instagram. Write every published page, README, comment, and interface label in English.

**Source code is required for every scene.** Publish the runnable scene source with its setup and run commands. Put a link to the source first on the scene page.

**Published paths are permanent.** Keep the names and locations of published files and folders. Update their contents in place and give new files new paths so existing links keep working.

## Add a reel

1. Use the published reel number and a short English slug, such as `03-scene-name`.
2. Export a representative vertical still from the actual reel as `preview.webp`. Put it in the reel folder and write descriptive English alt text.
3. Add `README.md` for GitHub and `index.html` for the public scene page. Include the preview, the Instagram reel link, and a short description with a link for each file. A short excerpt from the published source can serve as the link to the full file.
4. Copy the source for the published scene. Prefer one runnable file when it keeps the code clear. Include setup and a tested run command. Carry forward attribution where it applies.
5. Add a self-contained `model.html` or a browser demo when it is ready. Link to the Pages URL from the README to open it as a web page. Link to the GitHub file to show its source.
6. Once the source and run instructions are ready, add a card in the root `index.html` and an entry in the root `README.md`.
7. Test the source in a separate directory. Open the gallery, scene page, and model on a phone-width screen and follow their links.
8. Push to `main`, wait for GitHub Pages to deploy, then check the public links while signed out.

## Share a link

Use the [gallery](https://sapozhkov.github.io/ascii-lab-public/) when sharing the collection. Use the scene page when someone asks for a particular reel. For example, share the [clownfish page](https://sapozhkov.github.io/ascii-lab-public/01-clownfish/) for the code and model together.

## GitHub Pages

Use **Settings → Pages → Deploy from a branch → main → /(root)**. GitHub Pages serves the HTML as a website. The repository READMEs remain the entry points for people browsing the files on GitHub.
