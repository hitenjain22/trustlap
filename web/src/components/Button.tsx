import type { ButtonHTMLAttributes, ReactNode, Ref } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost'

const styles: Record<Variant, string> = {
  primary:
    'bg-[linear-gradient(180deg,#f9c766,var(--sodium)_55%,var(--sodium-2))] text-sodium-ink font-[650] shadow-[inset_0_1px_0_rgb(255_255_255/0.35),0_10px_24px_-12px_rgb(246_184_75/0.75)] hover:brightness-105',
  secondary:
    'border border-[var(--edge)] bg-[var(--glass)] text-paint hover:bg-[var(--glass-hover)] hover:border-line-strong font-[560] shadow-[inset_0_1px_0_rgb(255_255_255/0.05)]',
  ghost: 'text-text-2 hover:text-paint hover:bg-[var(--glass)] font-[560]',
}

export function Button({
  variant = 'secondary',
  className = '',
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant
  children: ReactNode
  ref?: Ref<HTMLButtonElement>
}) {
  return (
    <button
      type="button"
      className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-control px-4 text-[15px] transition-[color,background-color,border-color,transform,filter] duration-[120ms] active:translate-y-px disabled:cursor-not-allowed disabled:opacity-45 ${styles[variant]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  )
}
