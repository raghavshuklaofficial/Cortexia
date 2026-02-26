/**
 * Top header bar with status indicators.
 */

import { useEffect, useState } from "react";
import { Sun, Moon, Wifi, WifiOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useAppStore } from "@/lib/store";
import { cortexiaApi } from "@/lib/api";

export function Header() {
  const { theme, toggleTheme, isStreaming, fps } = useAppStore();
  const [apiStatus, setApiStatus] = useState<"online" | "offline" | "checking">("checking");

  useEffect(() => {
    const check = async () => {
      try {
        await cortexiaApi.health();
        setApiStatus("online");
      } catch {
        setApiStatus("offline");
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b bg-card/80 px-6 backdrop-blur-sm">
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-semibold text-foreground">
          Neural Face Intelligence
        </h1>
      </div>

      <div className="flex items-center gap-3">
        {/* Streaming indicator */}
        {isStreaming && (
          <Badge variant="success" className="animate-pulse-glow">
            <span className="mr-1.5 inline-block h-2 w-2 rounded-full bg-emerald-400" />
            LIVE · {fps} FPS
          </Badge>
        )}

        {/* API status */}
        <Badge variant={apiStatus === "online" ? "success" : "destructive"}>
          {apiStatus === "online" ? (
            <Wifi className="mr-1 h-3 w-3" />
          ) : (
            <WifiOff className="mr-1 h-3 w-3" />
          )}
          {apiStatus === "checking" ? "..." : apiStatus.toUpperCase()}
        </Badge>

        {/* Theme toggle */}
        <Button variant="ghost" size="icon" onClick={toggleTheme}>
          {theme === "dark" ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
        </Button>
      </div>
    </header>
  );
}
