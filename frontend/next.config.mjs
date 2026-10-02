/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  agentRules: false, // stop `next dev` from writing AGENTS.md / CLAUDE.md into the repo
};

export default nextConfig;
