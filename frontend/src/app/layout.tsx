import type { Metadata } from "next";
import Link from "next/link";
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
      <body className="bg-gray-950 text-gray-100 min-h-screen">
        <nav className="border-b border-gray-800 bg-gray-900">
          <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-8">
            <Link href="/" className="text-lg font-bold text-white">
              NBA Injury Risk
            </Link>
            <div className="flex gap-6 text-sm">
              <Link
                href="/"
                className="text-gray-400 hover:text-white transition"
              >
                Today
              </Link>
              <Link
                href="/performance"
                className="text-gray-400 hover:text-white transition"
              >
                Performance
              </Link>
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
