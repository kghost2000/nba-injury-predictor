import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NBA Injury Risk Predictions",
  description: "Daily injury risk predictions powered by machine learning",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
