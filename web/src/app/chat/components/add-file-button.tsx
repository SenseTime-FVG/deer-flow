/*
 * @Author: Qiutong Wei 
 * @Date: 2025-06-25 11:56:23 
 */


import { useRef } from "react";
import { AddIcon } from "~/app/chat/components/icons/addIcon"
import { ImageUp, FileUp } from "lucide-react"
import { Tooltip } from "~/components/deer-flow/tooltip";
import { Button } from "~/components/ui/button";
import { cn } from "~/lib/utils"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";

export function AddFileButton() {
    const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files ? event.target.files[0] : null;
        if (file) {
            // ToDo: Implement file upload logic
        }
    }
    const ImageInputRef = useRef<HTMLInputElement | null>(null);
    return (
        <>
            <DropdownMenu>
                <Tooltip className="max-w-40" title={
                    <div>
                        <p>This is a button that you can upload your file.</p>
                    </div>
                }>
                    <DropdownMenuTrigger asChild>
                        <Button
                            className={cn("h-10 w-10 rounded-full shrink-0")}
                            variant="outline"
                            size="icon">
                            <AddIcon />
                        </Button>
                    </DropdownMenuTrigger>
                </Tooltip>
                <DropdownMenuContent
                    side="top"
                    align="start">
                    <DropdownMenuItem onClick={() => {
                        ImageInputRef.current?.click();
                    }}>
                        <ImageUp className="text-current" />
                        <span>Upload Image</span>
                    </DropdownMenuItem>
                    <Tooltip className="max-w-40" title="Not Available Yet" side="bottom">
                        <DropdownMenuItem>
                            <FileUp className="text-current" />
                            <span>Upload File</span>
                        </DropdownMenuItem>
                    </Tooltip>
                </DropdownMenuContent>
            </DropdownMenu >
            <input type="file" ref={ImageInputRef} accept="image/*"
                onChange={handleImageChange} className="hidden" ></input>
        </>
    )
}