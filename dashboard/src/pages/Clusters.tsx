/**
 * Cluster discovery page — zero-shot identity grouping via HDBSCAN.
 */

import { useState } from "react";
import { Layers, Play, UserPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { cortexiaApi, type ClusterInfo } from "@/lib/api";

export default function ClustersPage() {
  const [clusters, setClusters] = useState<ClusterInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [minSize, setMinSize] = useState(5);
  const [discovered, setDiscovered] = useState(false);

  const runDiscovery = async () => {
    setLoading(true);
    try {
      const res = await cortexiaApi.discoverClusters(minSize);
      const data = res.data as { clusters?: ClusterInfo[] } | null;
      setClusters(data?.clusters || []);
      setDiscovered(true);
    } catch (err) {
      console.error("Cluster discovery failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const loadExisting = async () => {
    setLoading(true);
    try {
      const res = await cortexiaApi.listClusters();
      const data = res.data as { clusters?: ClusterInfo[] } | null;
      setClusters(data?.clusters || []);
      setDiscovered(true);
    } catch {
      // API not available
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Identity Clusters</h2>
          <p className="text-sm text-muted-foreground">
            Zero-shot identity discovery using HDBSCAN density clustering
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <label className="text-sm text-muted-foreground whitespace-nowrap">
              Min size:
            </label>
            <Input
              type="number"
              value={minSize}
              onChange={(e) => setMinSize(Number(e.target.value))}
              className="w-20"
              min={2}
            />
          </div>
          <Button onClick={runDiscovery} disabled={loading}>
            <Play className="mr-2 h-4 w-4" />
            {loading ? "Running..." : "Discover"}
          </Button>
          <Button variant="outline" onClick={loadExisting} disabled={loading}>
            Load Existing
          </Button>
        </div>
      </div>

      {!discovered ? (
        <Card>
          <CardContent className="flex h-64 flex-col items-center justify-center text-muted-foreground">
            <Layers className="mb-3 h-12 w-12 opacity-30" />
            <p className="text-lg font-medium">No Clusters Yet</p>
            <p className="text-sm">
              Click "Discover" to run HDBSCAN on all stored embeddings
            </p>
            <p className="mt-2 text-xs opacity-60">
              The algorithm will automatically find natural groupings without
              specifying the number of clusters
            </p>
          </CardContent>
        </Card>
      ) : clusters.length === 0 ? (
        <Card>
          <CardContent className="flex h-48 items-center justify-center text-muted-foreground">
            <p>No clusters found. Try lowering the minimum cluster size.</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {clusters.map((cluster) => (
            <Card key={cluster.id}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">
                    Cluster #{cluster.id}
                  </CardTitle>
                  <Badge variant="secondary">
                    {cluster.member_count} members
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <div className="flex -space-x-2">
                      {Array.from({ length: Math.min(cluster.member_count, 5) }).map(
                        (_, i) => (
                          <div
                            key={i}
                            className="flex h-8 w-8 items-center justify-center rounded-full border-2 border-card bg-cortexia-500/20 text-xs text-cortexia-400"
                          >
                            {i + 1}
                          </div>
                        )
                      )}
                      {cluster.member_count > 5 && (
                        <div className="flex h-8 w-8 items-center justify-center rounded-full border-2 border-card bg-muted text-xs text-muted-foreground">
                          +{cluster.member_count - 5}
                        </div>
                      )}
                    </div>
                  </div>
                  <Button variant="outline" size="sm" className="w-full">
                    <UserPlus className="mr-2 h-3 w-3" />
                    Assign to Identity
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
