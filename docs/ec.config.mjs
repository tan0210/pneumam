// @ts-check
import { defineEcConfig } from "@astrojs/starlight/expressive-code";

// The code colours match the landing page: keyword coral, function blue, string green, number amber.
const pneumamCode = {
	name: 'pneumam',
	type: 'dark',
	colors: { 'editor.background': '#0d0f15', 'editor.foreground': '#d7dbe4' },
	tokenColors: [
		{ scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: '#a3aab8', fontStyle: 'italic' } },
		{ scope: ['keyword', 'storage', 'keyword.control', 'constant.language'], settings: { foreground: '#ff8f6b' } },
		{ scope: ['entity.name.function', 'support.function', 'meta.function-call.generic', 'entity.name.type', 'support.class'], settings: { foreground: '#7fb4ff' } },
		{ scope: ['string', 'string.quoted'], settings: { foreground: '#a5e3a0' } },
		{ scope: ['constant.numeric'], settings: { foreground: '#f5c46b' } },
		{ scope: ['variable.parameter', 'support.type.property-name', 'variable.other.property'], settings: { foreground: '#c7cedb' } },
		{ scope: ['entity.name.section', 'entity.name.table'], settings: { foreground: '#7fb4ff' } },
	],
};

export default defineEcConfig({
				themes: [pneumamCode],
				styleOverrides: { frames: { terminalTitlebarBackground: '#13151c', terminalTitlebarBorderBottomColor: '#22252f', terminalTitlebarDotsForeground: '#3a3e4a', terminalBackground: '#0d0f15', editorTabBarBackground: '#13151c', editorActiveTabBackground: '#0d0f15', frameBoxShadowCssValue: 'none' }, borderRadius: '6px', borderColor: '#22252f', codeFontFamily: "'JetBrains Mono', ui-monospace, monospace", codeFontSize: '14px', uiFontFamily: "'Mona Sans', system-ui, sans-serif" },
			});
