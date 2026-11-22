import type { ComponentProps } from "react"
import { X } from "lucide-react"
import { Dialog as DialogPrimitive } from "radix-ui"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const Dialog = DialogPrimitive.Root
const DialogTrigger = DialogPrimitive.Trigger
const DialogClose = DialogPrimitive.Close
const DialogTitle = DialogPrimitive.Title
const DialogDescription = DialogPrimitive.Description

function DialogContent({ className, children, ...props }: ComponentProps<typeof DialogPrimitive.Content>) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="dialog-overlay" />
      <DialogPrimitive.Content className={cn("dialog-content corner-frame", className)} {...props}>
        {children}
        <DialogClose asChild>
          <Button className="dialog-close" variant="ghost" size="icon" aria-label="Close">
            <X />
          </Button>
        </DialogClose>
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  )
}

export { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle, DialogTrigger }
