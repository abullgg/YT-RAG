import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Allow cross-origin requests to the FastAPI backend in dev
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000'}/:path*`,
      },
    ];
  },
};

export default nextConfig;
