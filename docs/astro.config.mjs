// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLlmsTxt from 'starlight-llms-txt';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Workaround: on Windows, Expressive Code 0.44.3 links `_astro/ec.<hash>.css` with a different
 * hash than the file it writes, so code blocks lose their styles. After the build, point every
 * page at the stylesheet that exists. Remove once Expressive Code fixes it.
 * @returns {import('astro').AstroIntegration}
 */
function fixCodeStylesheet() {
	return {
		name: 'pneumam:fix-ec-stylesheet',
		hooks: {
			'astro:build:done': async ({ dir, logger }) => {
				const out = fileURLToPath(dir);
				const css = (await readdir(join(out, '_astro'))).find((f) => /^ec\.[\w-]+\.css$/.test(f));
				if (!css) return;
				let fixed = 0;
				for (const f of await readdir(out, { recursive: true })) {
					if (!f.endsWith('.html')) continue;
					const path = join(out, f), html = await readFile(path, 'utf8');
					const next = html.replace(/_astro\/ec\.[\w-]+\.css/g, `_astro/${css}`);
					if (next !== html) { await writeFile(path, next); fixed++; }
				}
				if (fixed) logger.info(`pointed ${fixed} pages at _astro/${css}`);
			},
		},
	};
}

/**
 * Prefix root-absolute links in the built pages with the base path, for hosting under a
 * subpath such as GitHub Pages' /pneumam/. Markdown links, LinkCard hrefs and hero actions
 * are written as /page/, and Starlight does not add the base to them.
 * @param {string | undefined} base
 * @returns {import('astro').AstroIntegration}
 */
function prefixBase(base) {
	return {
		name: 'pneumam:prefix-base',
		hooks: {
			'astro:build:done': async ({ dir, logger }) => {
				const root = (base ?? '').replace(/\/+$/, '');
				if (!root) return;
				const out = fileURLToPath(dir);
				const name = root.slice(1).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
				// a single leading slash, not // and not a link that already starts with the base
				const link = new RegExp(`((?:href|src)=["'])/(?!/|${name}(?:/|["']))`, 'g');
				let fixed = 0;
				for (const f of await readdir(out, { recursive: true })) {
					if (!f.endsWith('.html')) continue;
					const path = join(out, f), html = await readFile(path, 'utf8');
					const next = html.replace(link, `$1${root}/`);
					if (next !== html) { await writeFile(path, next); fixed++; }
				}
				if (fixed) logger.info(`prefixed links with ${root} in ${fixed} pages`);
			},
		},
	};
}

export default defineConfig({
	// The public address of the docs. llms.txt needs it for absolute links.
	// Set SITE_URL when building for the real domain; the placeholder is never a live site.
	site: process.env.SITE_URL ?? 'https://pneumam.example',
	// The subpath the docs are served under, such as /pneumam on GitHub Pages. Unset for a root domain.
	base: process.env.BASE_PATH,
	integrations: [
		starlight({
			title: 'Pneumam',
			description: 'Fast inference and training for the Anima image model.',
			// llms.txt, llms-full.txt and llms-small.txt are generated from these pages on every build.
			plugins: [
				starlightLlmsTxt({
					projectName: 'Pneumam',
					description: 'Pneumam generates Anima images in ComfyUI, from the command line and from Python, about twice as fast as ComfyUI\'s own Anima support. It also trains LoRA and LLLite adapters.',
					details: [
						'- Install it with `uv add git+https://github.com/tan0210/pneumam`, plus the extra for the GPU: `sm120` for the RTX 50 series and RTX PRO 6000 Blackwell, `sm90` for the H100 and H200, `sm89` for the RTX 40 series, L4 and L40, `sm86` for the RTX 30 series, A10 and A40.',
						'- A100 and older GPUs are not supported. RTX 30 series cards run without FP8.',
						'- Convert the model once with `python scripts/convert.py --out checkpoints` before the first run.',
						'- ComfyUI must be started with `--highvram --disable-cuda-malloc`.',
					].join('\n'),
					// pages in reading order (the sidebar's), not alphabetical
					promote: [
						'index*', 'start/install', 'start/convert', 'comfyui/nodes', 'comfyui/remote',
						'generate/command-line', 'generate/python', 'generate/sampling',
						'adapters/lora-lllite', 'adapters/training', 'reference/gpus', 'reference/environment',
					],
					demote: ['reference/speed'],
					// the small file keeps the how-to pages; the front page and the speed figures are in the full file only
					exclude: ['index', 'reference/speed'],
					// Starlight's "Section titled …" heading links are noise to a model
					customSelectors: { all: ['.sl-anchor-link'] },
					// keep line breaks so tables and lists stay readable; asides stay in, since notes carry GPU support
					minify: { whitespace: false, note: false, tip: false },
					optionalLinks: [{ label: 'Source on GitHub', url: 'https://github.com/tan0210/pneumam', description: 'The code, the README and the issue tracker.' }],
				}),
			],
			favicon: '/favicon.svg',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/tan0210/pneumam' }],
			editLink: { baseUrl: 'https://github.com/tan0210/pneumam/edit/main/docs/' },
			customCss: ['./src/styles/pneumam.css'],
			head: [
				{ tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.googleapis.com' } },
				{ tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' } },
				{ tag: 'link', attrs: { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Mona+Sans:wdth,wght@75..125,400..600&family=JetBrains+Mono:wght@400..600&display=swap' } },
				// Command tabs (<Tabs syncKey="os">) open on Windows for Windows visitors, else on Linux (Bash).
				// A tab the visitor picks is stored by Starlight and wins over this guess.
				{
					tag: 'script',
					content: `try{var k='starlight-synced-tabs__os',p=(navigator.userAgentData&&navigator.userAgentData.platform)||navigator.platform||navigator.userAgent;if(!localStorage.getItem(k)&&/win/i.test(p))localStorage.setItem(k,'Windows')}catch(e){}`,
				},
			],
			sidebar: [
				{ label: 'Start', items: [{ autogenerate: { directory: 'start' } }] },
				{ label: 'Generate', items: [{ autogenerate: { directory: 'generate' } }] },
				{ label: 'ComfyUI', items: [{ autogenerate: { directory: 'comfyui' } }] },
				{ label: 'Adapters', items: [{ autogenerate: { directory: 'adapters' } }] },
				{ label: 'Reference', items: [{ autogenerate: { directory: 'reference' } }] },
			],
		}),
		fixCodeStylesheet(),
		prefixBase(process.env.BASE_PATH),
	],
});
