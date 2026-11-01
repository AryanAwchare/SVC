import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VocalMorph - Zero-Shot Singing Voice Conversion",
  description: "Convert any song into your singing voice using Seed-VC and neural stem separation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 antialiased selection:bg-indigo-600 selection:text-white">
        {children}
      </body>
    </html>
  );
}
