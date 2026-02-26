/**
 * Sidebar navigation component.
 */

import { NavLink } from "react-router-dom";
import {
  Brain,
  Camera,
  Users,
  Activity,
  Shield,
  BarChart3,
  Layers,
  Settings,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/lib/store";
import { Button } from "@/components/ui/button";

const navigation = [
  { name: "Live Feed", href: "/", icon: Camera },
  { name: "Identities", href: "/identities", icon: Users },
  { name: "Recognition Log", href: "/events", icon: Activity },
  { name: "Analytics", href: "/analytics", icon: BarChart3 },
  { name: "Clusters", href: "/clusters", icon: Layers },
  { name: "Forensics", href: "/forensics", icon: Shield },
  { name: "Settings", href: "/settings", icon: Settings },
];

export function Sidebar() {
  const { sidebarOpen, toggleSidebar } = useAppStore();

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 h-screen border-r bg-card transition-all duration-300",
        sidebarOpen ? "w-64" : "w-16"
      )}
    >
      {/* Logo */}
      <div className="flex h-16 items-center justify-between border-b px-4">
        {sidebarOpen && (
          <div className="flex items-center gap-2">
            <Brain className="h-7 w-7 text-cortexia-500" />
            <span className="text-lg font-bold tracking-wide">CORTEXIA</span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleSidebar}
          className={cn(!sidebarOpen && "mx-auto")}
        >
          {sidebarOpen ? (
            <ChevronLeft className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Nav Links */}
      <nav className="flex flex-col gap-1 p-2">
        {navigation.map((item) => (
          <NavLink
            key={item.href}
            to={item.href}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                isActive
                  ? "bg-cortexia-500/10 text-cortexia-400"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
                !sidebarOpen && "justify-center px-2"
              )
            }
          >
            <item.icon className="h-5 w-5 flex-shrink-0" />
            {sidebarOpen && <span>{item.name}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Version tag */}
      {sidebarOpen && (
        <div className="absolute bottom-4 left-0 right-0 px-4">
          <div className="rounded-lg bg-muted p-3 text-center text-xs text-muted-foreground">
            <span className="font-mono">v1.0.0</span>
            <span className="mx-1">·</span>
            <span>Neural Engine</span>
          </div>
        </div>
      )}
    </aside>
  );
}
