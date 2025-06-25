/*
 * @Author: Qiutong Wei 
 * @Date: 2025-06-25 16:34:37 
 */


import type { Resource } from "~/core/messages";

interface FileContainerProps {
  resources: Resource[];
}

export function FileContainer({ resources }: FileContainerProps) {
  const imageCount = resources.filter((r) => r.title.startsWith("image/")).length;
  const fileCount = resources.length - imageCount;

  return (
    <div className="text-sm text-muted-foreground px-4 py-2 space-x-4">
      {resources.length === 0 ? (
        <span>No files uploaded.</span>
      ) : (
        <>
          <span>🖼 Images: {imageCount}</span>
          <span>📁 Files: {fileCount}</span>
        </>
      )}
    </div>
  );
}
