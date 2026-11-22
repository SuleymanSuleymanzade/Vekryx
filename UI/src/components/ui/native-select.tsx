import type { ComponentProps } from "react"
import { ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"

export function NativeSelect({ className, ...props }: ComponentProps<"select">) {
  return (
    <span className="select-wrap">
      <select className={cn("field-control", className)} {...props} />
      <ChevronDown aria-hidden="true" />
    </span>
  )
}
