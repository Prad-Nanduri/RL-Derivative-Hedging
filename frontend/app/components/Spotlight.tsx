"use client";

import { useEffect } from "react";

export default function Spotlight() {
  useEffect(() => {
    const el = document.documentElement;
    const onMove = (e: PointerEvent) => {
      el.style.setProperty("--mx", `${e.clientX}px`);
      el.style.setProperty("--my", `${e.clientY}px`);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, []);
  return <div className="spotlight" aria-hidden="true" />;
}
