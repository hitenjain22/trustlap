import type { ButtonHTMLAttributes, ReactNode, Ref } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost'

const styles: Record<Variant, string> = {
  primary: 'bg-sodium text-sodium-ink hover:bg-[#f5c35f] font-[650]',
  secondary: 'border border-line bg-asphalt-800 text-paint hover:bg-asphalt-700 font-[560]',
  ghost: 'text-text-2 hover:text-paint hover:bg-asphalt-700 font-[560]',
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
      className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-control px-4 text-[15px] transition-colors duration-[120ms] disabled:cursor-not-allowed disabled:opacity-45 ${styles[variant]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  )
}
