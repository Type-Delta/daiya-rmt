import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'var(--bg)',
        surface: 'var(--surface)',
        raised: 'var(--raised)',
        edge: 'var(--edge)',
        ink: 'var(--ink)',
        muted: 'var(--muted)',
        faint: 'var(--faint)',
        primary: 'var(--primary)',
        'primary-ink': 'var(--primary-ink)',
        accent: 'var(--accent)',
        danger: 'var(--danger)',
        warn: 'var(--warn)',
        ok: 'var(--ok)',
      },
      fontFamily: {
        // System stacks on purpose: best native rendering for Thai + Japanese.
        sans: [
          'system-ui',
          '-apple-system',
          '"Segoe UI"',
          'Roboto',
          '"Noto Sans"',
          '"Noto Sans Thai"',
          '"Noto Sans JP"',
          '"Hiragino Sans"',
          '"Yu Gothic UI"',
          'sans-serif',
        ],
        mono: [
          'ui-monospace',
          '"Cascadia Mono"',
          '"SF Mono"',
          'Menlo',
          'Consolas',
          'monospace',
        ],
      },
    },
  },
} satisfies Config;
