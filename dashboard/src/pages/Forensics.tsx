/**
 * Forensic analysis page — deep liveness & spoofing detection.
 */

import { useRef, useState } from "react";
import { Shield, FileImage, CheckCircle2, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cortexiaApi, type ForensicAnalysisResult } from "@/lib/api";

export default function ForensicsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<ForensicAnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"liveness" | "full">("liveness");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (f: File) => {
    setFile(f);
    setResult(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const res =
        mode === "liveness"
          ? await cortexiaApi.analyzeLiveness(file)
          : await cortexiaApi.forensicAnalyze(file);
      setResult(res);
    } catch (err) {
      console.error("Analysis failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const getVerdictColor = (verdict: string) => {
    switch (verdict.toUpperCase()) {
      case "LIVE":
        return "success";
      case "SPOOF":
        return "destructive";
      default:
        return "warning";
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Forensic Analysis</h2>
        <p className="text-sm text-muted-foreground">
          Deep anti-spoofing inspection with multi-spectral analysis
        </p>
      </div>

      <Tabs value={mode} onValueChange={(v) => setMode(v as "liveness" | "full")}>
        <TabsList>
          <TabsTrigger value="liveness">Liveness Check</TabsTrigger>
          <TabsTrigger value="full">Full Forensic Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value={mode}>
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Upload panel */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Input Image</CardTitle>
                <CardDescription>
                  Upload a face image for{" "}
                  {mode === "liveness" ? "liveness verification" : "full forensic analysis"}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  className="cursor-pointer rounded-lg border-2 border-dashed p-8 text-center transition-colors hover:border-cortexia-500/50 hover:bg-muted/30"
                  onClick={() => inputRef.current?.click()}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    const droppedFile = e.dataTransfer.files[0];
                    if (droppedFile) handleFileSelect(droppedFile);
                  }}
                >
                  {preview ? (
                    <img
                      src={preview}
                      alt="Upload preview"
                      className="mx-auto max-h-64 rounded-lg object-contain"
                    />
                  ) : (
                    <>
                      <FileImage className="mx-auto mb-3 h-12 w-12 text-muted-foreground opacity-30" />
                      <p className="text-sm text-muted-foreground">
                        Drop an image here or click to upload
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground opacity-60">
                        JPEG, PNG up to 10MB
                      </p>
                    </>
                  )}
                </div>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files?.[0]) handleFileSelect(e.target.files[0]);
                  }}
                />
                <Button
                  className="mt-4 w-full"
                  onClick={analyze}
                  disabled={!file || loading}
                >
                  <Shield className="mr-2 h-4 w-4" />
                  {loading ? "Analyzing..." : "Run Analysis"}
                </Button>
              </CardContent>
            </Card>

            {/* Results panel */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Analysis Results</CardTitle>
              </CardHeader>
              <CardContent>
                {!result ? (
                  <div className="flex h-64 flex-col items-center justify-center text-muted-foreground">
                    <Shield className="mb-3 h-10 w-10 opacity-30" />
                    <p>Upload an image and click "Run Analysis"</p>
                  </div>
                ) : !result.face_detected ? (
                  <div className="flex h-64 flex-col items-center justify-center text-muted-foreground">
                    <XCircle className="mb-3 h-10 w-10 opacity-30" />
                    <p>No face detected in the image</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Processing Time</span>
                      <span className="font-mono">{result.processing_time_ms?.toFixed(0) ?? 0} ms</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Trust Score</span>
                      <span className="font-mono">{((result.trust_score ?? 0) * 100).toFixed(0)}%</span>
                    </div>

                    {result.liveness && (
                      <div className="rounded-lg border bg-muted/30 p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">Liveness</span>
                          <Badge variant={getVerdictColor(result.liveness.verdict) as "success" | "destructive" | "warning"}>
                            {result.liveness.verdict.toUpperCase() === "LIVE" ? (
                              <CheckCircle2 className="mr-1 h-3 w-3" />
                            ) : (
                              <XCircle className="mr-1 h-3 w-3" />
                            )}
                            {result.liveness.verdict.toUpperCase()}
                          </Badge>
                        </div>

                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">Confidence</span>
                            <span className="font-mono">
                              {(result.liveness.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-2 rounded-full bg-muted">
                            <div
                              className={`h-full rounded-full transition-all ${
                                result.liveness.verdict.toUpperCase() === "LIVE"
                                  ? "bg-emerald-500"
                                  : result.liveness.verdict.toUpperCase() === "SPOOF"
                                  ? "bg-red-500"
                                  : "bg-amber-500"
                              }`}
                              style={{ width: `${result.liveness.confidence * 100}%` }}
                            />
                          </div>
                        </div>

                        <div className="flex items-center justify-between text-xs border-t pt-2">
                          <span className="text-muted-foreground">Method</span>
                          <span className="font-mono capitalize">{result.liveness.method}</span>
                        </div>
                      </div>
                    )}

                    {result.face_quality_score !== undefined && result.face_quality_score > 0 && (
                      <div className="flex items-center justify-between text-xs border-t pt-2">
                        <span className="text-muted-foreground">Image Quality</span>
                        <Badge variant={result.face_quality_score > 0.7 ? "success" : "warning"}>
                          {(result.face_quality_score * 100).toFixed(0)}%
                        </Badge>
                      </div>
                    )}

                    {result.attributes && (
                      <div className="rounded-lg border bg-muted/30 p-4 space-y-2">
                        <span className="text-sm font-medium">Attributes</span>
                        {result.attributes.age != null && (
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">Age</span>
                            <span className="font-mono">{result.attributes.age}</span>
                          </div>
                        )}
                        {result.attributes.gender && (
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">Gender</span>
                            <span className="font-mono capitalize">
                              {result.attributes.gender}
                              {result.attributes.gender_confidence != null &&
                                ` (${(result.attributes.gender_confidence * 100).toFixed(0)}%)`}
                            </span>
                          </div>
                        )}
                        {result.attributes.emotion && (
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">Emotion</span>
                            <span className="font-mono capitalize">
                              {result.attributes.emotion}
                              {result.attributes.emotion_confidence != null &&
                                ` (${(result.attributes.emotion_confidence * 100).toFixed(0)}%)`}
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
