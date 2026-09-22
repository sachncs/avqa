import type { ReactNode } from "react";

type Props = {
  children: ReactNode;
  className?: string;
  delay?: number;
  y?: number;
  id?: string;
};

export function FadeIn({ children, className, id }: Props) {
  return (
    <div id={id} className={`min-w-0 ${className ?? ""}`}>
      {children}
    </div>
  );
}
