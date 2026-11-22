import type { ComponentProps } from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

const buttonVariants = cva("button", {
  variants: {
    variant: { outline: "button-outline", ghost: "button-ghost", primary: "button-primary" },
    size: { default: "", icon: "button-icon" },
  },
  defaultVariants: { variant: "outline", size: "default" },
})

function Button({
  className, variant, size, asChild = false, type = "button", ...props
}: ComponentProps<"button"> & VariantProps<typeof buttonVariants> & { asChild?: boolean }) {
  const Component = asChild ? Slot.Root : "button"

  return (
    <Component
      data-slot="button"
      className={cn(buttonVariants({ variant, size }), className)}
      type={asChild ? undefined : type}
      {...props}
    />
  )
}

export { Button }
