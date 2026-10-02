import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Agentic OSINT Analyst",
  description:
    "Retrieval-grounded reports on entity categories from public records (OFAC, SEC EDGAR, CourtListener), with the source excerpts shown.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
