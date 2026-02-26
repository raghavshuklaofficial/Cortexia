/**
 * Settings page — API configuration and system info.
 */

import { useEffect, useState } from "react";
import { Server, Cpu, HardDrive } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cortexiaApi, type ReadinessStatus } from "@/lib/api";

export default function SettingsPage() {
  const [readiness, setReadiness] = useState<ReadinessStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const ready = await cortexiaApi.ready();
        setReadiness(ready);
      } catch {
        // API not available
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Settings</h2>
        <p className="text-sm text-muted-foreground">
          System configuration and status
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* System Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Server className="h-4 w-4" />
              System Status
            </CardTitle>
            <CardDescription>Component health checks</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <p className="text-sm text-muted-foreground">Checking...</p>
            ) : !readiness ? (
              <div className="space-y-2">
                <Badge variant="destructive">API Unreachable</Badge>
                <p className="text-sm text-muted-foreground">
                  The API server is not responding. Make sure the backend is running.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Overall</span>
                  <Badge
                    variant={readiness.status === "ready" ? "success" : "destructive"}
                  >
                    {readiness.status.toUpperCase()}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Database</span>
                  <Badge variant={readiness.database === "connected" ? "success" : "destructive"}>
                    {readiness.database === "connected" ? "OK" : "FAIL"}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Redis</span>
                  <Badge variant={readiness.redis === "connected" ? "success" : "destructive"}>
                    {readiness.redis === "connected" ? "OK" : "FAIL"}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">ML Models</span>
                  <Badge variant={readiness.models_loaded ? "success" : "destructive"}>
                    {readiness.models_loaded ? "OK" : "FAIL"}
                  </Badge>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* About */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Cpu className="h-4 w-4" />
              About CORTEXIA
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Version</span>
              <span className="font-mono">1.0.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Codename</span>
              <span>Neural Face Intelligence</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Detector</span>
              <span>RetinaFace (InsightFace)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Embedder</span>
              <span>ArcFace buffalo_l (512-d)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Anti-Spoof</span>
              <span>Multi-spectral Ensemble</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Clustering</span>
              <span>HDBSCAN</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Calibration</span>
              <span>Platt Scaling</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Vector Store</span>
              <span>pgvector (IVFFlat)</span>
            </div>
          </CardContent>
        </Card>

        {/* Architecture */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <HardDrive className="h-4 w-4" />
              Trust Pipeline Architecture
            </CardTitle>
            <CardDescription>
              Every face passes through the complete trust pipeline
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              {[
                { label: "Detection", desc: "RetinaFace" },
                { label: "Alignment", desc: "ArcFace 112x112" },
                { label: "Liveness", desc: "FFT + Color + Texture + Moire" },
                { label: "Embedding", desc: "512-d L2-norm" },
                { label: "Recognition", desc: "Cosine + Platt" },
                { label: "Attributes", desc: "Age / Gender / Emotion" },
              ].map((step, i) => (
                <div key={step.label} className="flex items-center gap-2">
                  <div className="rounded-lg border bg-muted/50 px-3 py-2 text-center">
                    <p className="font-medium">{step.label}</p>
                    <p className="text-muted-foreground">{step.desc}</p>
                  </div>
                  {i < 5 && (
                    <span className="text-muted-foreground">&rarr;</span>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
