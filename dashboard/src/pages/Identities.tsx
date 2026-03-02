/**
 * Identity management page — CRUD for enrolled faces.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  UserPlus,
  Search,
  Trash2,
  Upload,
  ChevronLeft,
  ChevronRight,
  User,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cortexiaApi, type Identity } from "@/lib/api";
import { useAppStore } from "@/lib/store";

export default function IdentitiesPage() {
  const [identities, setLocalIdentities] = useState<Identity[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newFile, setNewFile] = useState<File | null>(null);
  const fileInputRefs = useRef<Map<number, HTMLInputElement>>(new Map());
  const { setIdentities: setGlobalIdentities } = useAppStore();

  const pageSize = 12;

  const loadIdentities = useCallback(async () => {
    setLoading(true);
    try {
      const res = await cortexiaApi.listIdentities(page, pageSize, search || undefined);
      setLocalIdentities(res.identities);
      setTotal(res.pagination.total);
      setGlobalIdentities(res.identities);
    } catch {
      // API not available
    } finally {
      setLoading(false);
    }
  }, [page, search, setGlobalIdentities]);

  useEffect(() => {
    loadIdentities();
  }, [loadIdentities]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    try {
      await cortexiaApi.createIdentity(newName, newFile || undefined);
      setNewName("");
      setNewFile(null);
      setShowCreate(false);
      loadIdentities();
    } catch (err) {
      console.error("Failed to create identity:", err);
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm("Delete this identity? This action cannot be undone.")) return;
    try {
      await cortexiaApi.deleteIdentity(id);
      loadIdentities();
    } catch (err) {
      console.error("Failed to delete identity:", err);
    }
  };

  const handleAddFaces = async (id: number, files: FileList) => {
    try {
      await cortexiaApi.addFaces(id, files);
      loadIdentities();
    } catch (err) {
      console.error("Failed to add faces:", err);
    }
  };

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Identity Gallery</h2>
          <p className="text-sm text-muted-foreground">
            {total} enrolled {total === 1 ? "identity" : "identities"}
          </p>
        </div>
        <Button onClick={() => setShowCreate(true)}>
          <UserPlus className="mr-2 h-4 w-4" /> Enroll Identity
        </Button>
      </div>

      {/* Search */}
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search identities..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
          className="pl-10"
        />
      </div>

      {/* Create dialog (inline) */}
      {showCreate && (
        <Card className="border-cortexia-500/30">
          <CardHeader>
            <CardTitle className="text-base">Enroll New Identity</CardTitle>
          </CardHeader>
          <CardContent className="flex items-end gap-4">
            <div className="flex-1 space-y-2">
              <label className="text-sm font-medium">Name</label>
              <Input
                placeholder="Enter name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Face Image (optional)</label>
              <Input
                type="file"
                accept="image/*"
                onChange={(e) => setNewFile(e.target.files?.[0] || null)}
              />
            </div>
            <Button onClick={handleCreate} disabled={!newName.trim()}>
              Create
            </Button>
            <Button variant="outline" onClick={() => setShowCreate(false)}>
              Cancel
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Identity Grid */}
      {loading ? (
        <div className="flex h-48 items-center justify-center text-muted-foreground">
          Loading...
        </div>
      ) : identities.length === 0 ? (
        <Card>
          <CardContent className="flex h-48 flex-col items-center justify-center text-muted-foreground">
            <User className="mb-3 h-10 w-10 opacity-30" />
            <p>No identities enrolled yet</p>
            <p className="text-xs">Click "Enroll Identity" to get started</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {identities.map((identity) => (
            <Card key={identity.id} className="group relative overflow-hidden">
              <CardContent className="p-4">
                <div className="mb-3 flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-cortexia-500/10 text-cortexia-400">
                    <User className="h-6 w-6" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="truncate font-medium">{identity.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {identity.face_count} face{identity.face_count !== 1 ? "s" : ""}
                    </p>
                  </div>
                </div>

                <div className="space-y-1.5 text-xs text-muted-foreground">
                  <div className="flex justify-between">
                    <span>Privacy Score</span>
                    <span className="font-mono">
                      {(identity.privacy_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  {identity.last_seen && (
                    <div className="flex justify-between">
                      <span>Last Seen</span>
                      <span>{new Date(identity.last_seen).toLocaleDateString()}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span>Enrolled</span>
                    <span>{new Date(identity.created_at).toLocaleDateString()}</span>
                  </div>
                </div>

                <div className="mt-3 flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                    onClick={() => fileInputRefs.current.get(identity.id)?.click()}
                  >
                    <Upload className="mr-1 h-3 w-3" />
                    Add Faces
                  </Button>
                  <input
                    ref={(el) => {
                      if (el) fileInputRefs.current.set(identity.id, el);
                      else fileInputRefs.current.delete(identity.id);
                    }}
                    type="file"
                    multiple
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files) handleAddFaces(identity.id, e.target.files);
                    }}
                  />
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-destructive hover:text-destructive"
                    onClick={() => handleDelete(identity.id)}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm text-muted-foreground">
            Page {page} of {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
