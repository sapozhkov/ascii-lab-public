# Publishing guide

This repository shares the source code behind ASCII Lab reels and provides public links for Instagram. Write every published page, README, comment, and interface label in English.

**Source code is required for every scene.** Publish the runnable scene source with its setup and run commands. Put a link to the source first on the scene page.

**Published paths are permanent.** Keep the names and locations of published files and folders. Update their contents in place and give new files new paths so existing links keep working.

## Add a reel

1. Use the published reel number and a short English slug, such as `03-scene-name`.
2. Export a representative vertical still from the actual reel as `preview.webp`. Write descriptive English alt text. Keep the full still on the scene page and show it as a compact square tile in the gallery. Reuse the existing grid, which fits two tiles per row on phones and up to four on wider screens.
3. Add `README.md` for GitHub and `index.html` for the public scene page. Put the source card first. Show a short excerpt from the actual published file and make the entire card open that file on GitHub. Put the model card below the code. Keep links to the run instructions and the Instagram reel visible.
4. Copy the runnable source for the published scene. Prefer one file when it keeps the code clear. If the scene uses multiple modules, link to the entry point and explain what each module does. List the dependencies, provide tested run commands, and carry forward attribution where it applies.
5. When the scene has a model page, include `model.html` and its assets. Use the same dark palette for the model and its preview. Choose a representative frame from the diagram, such as the top view, and make the whole preview card open the model. Crop the image to match the height of the code card. Keep the key subject visible and let the code keep its natural height. Use the Pages URL in the README.
6. Add the scene to the root `index.html` and `README.md` once the source and run instructions are ready.
7. Run the source from a separate directory and check a representative frame against the working render. If the source exports PNG and MP4, test both. On desktop and at phone width, check the preview crops, code and model card heights, and navigation. For interactive models, test play, pause, and seeking. Compare the diagram with the reel at the same times.
8. Add the actual Instagram reel link before publishing. Push to `main`, wait for GitHub Pages to deploy, and open the gallery, scene page, source, and any model page while signed out. Share the gallery or scene URL on Instagram after this check.

## Model pages

Generate the diagram from the same simulation used for the reel. Keep one timeline for the diagram and the ASCII view. Pick the projections, key moments, and readouts that explain how that scene works.

The [clownfish model](https://sapozhkov.github.io/ascii-lab-public/01-clownfish/model.html) shows the route, forces, and camera. The [jellyfish model](https://sapozhkov.github.io/ascii-lab-public/02-jellyfish-heart/model.html) pairs a silent cut of the reel with the same frame of the diagram. It lets viewers compare the swimming geometry with the final pose.

Commit a theme change and its matching preview together, separately from changes to the source or the model data. That makes the appearance easy to revert.

## Share a link

Use the [gallery](https://sapozhkov.github.io/ascii-lab-public/) when sharing the collection. Use the scene page when someone asks for a particular reel. For example, share the [clownfish page](https://sapozhkov.github.io/ascii-lab-public/01-clownfish/) for the code and model together.

## GitHub Pages

Use **Settings → Pages → Deploy from a branch → main → /(root)**. GitHub Pages serves the HTML as a website. The repository READMEs remain the entry points for people browsing the files on GitHub.
