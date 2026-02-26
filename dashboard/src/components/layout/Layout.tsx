import { Outlet } from "react-router-dom";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/lib/store";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";

export function Layout() {
  const { sidebarOpen } = useAppStore();

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <div
        className={cn(
          "transition-all duration-300",
          sidebarOpen ? "ml-64" : "ml-16"
        )}
      >
        <Header />
        <main className="p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
