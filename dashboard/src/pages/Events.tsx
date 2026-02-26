/**
 * Recognition event log — filterable audit trail.
 */

import { useCallback, useEffect, useState } from "react";
import { Activity, Filter, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cortexiaApi, type RecognitionEvent } from "@/lib/api";

export default function EventsPage() {
  const [events, setEvents] = useState<RecognitionEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [filterSource, setFilterSource] = useState("");
  const [filterKnown, setFilterKnown] = useState<boolean | undefined>();
  const [filterSpoof, setFilterSpoof] = useState<boolean | undefined>();

  const pageSize = 20;

  const loadEvents = useCallback(async () => {
    setLoading(true);
    try {
      const res = await cortexiaApi.listEvents({
        page,
        size: pageSize,
        source: filterSource || undefined,
        is_known: filterKnown,
        is_spoof: filterSpoof,
      });
      setEvents(res.events);
      setTotal(res.pagination.total);
    } catch {
      // API not available
    } finally {
      setLoading(false);
    }
  }, [page, filterSource, filterKnown, filterSpoof]);

  useEffect(() => {
    loadEvents();
  }, [loadEvents]);

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Recognition Log</h2>
        <p className="text-sm text-muted-foreground">
          Forensic audit trail of all recognition events ({total} total)
        </p>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="flex flex-wrap items-center gap-4 p-4">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Filter by source..."
            value={filterSource}
            onChange={(e) => {
              setFilterSource(e.target.value);
              setPage(1);
            }}
            className="max-w-xs"
          />
          <div className="flex gap-2">
            <Button
              variant={filterKnown === true ? "default" : "outline"}
              size="sm"
              onClick={() => setFilterKnown(filterKnown === true ? undefined : true)}
            >
              Known
            </Button>
            <Button
              variant={filterKnown === false ? "default" : "outline"}
              size="sm"
              onClick={() => setFilterKnown(filterKnown === false ? undefined : false)}
            >
              Unknown
            </Button>
            <Button
              variant={filterSpoof === true ? "destructive" : "outline"}
              size="sm"
              onClick={() => setFilterSpoof(filterSpoof === true ? undefined : true)}
            >
              Spoofs Only
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Events table */}
      <Card>
        <CardContent className="p-0">
          {loading ? (
            <div className="flex h-48 items-center justify-center text-muted-foreground">
              Loading...
            </div>
          ) : events.length === 0 ? (
            <div className="flex h-48 flex-col items-center justify-center text-muted-foreground">
              <Activity className="mb-3 h-10 w-10 opacity-30" />
              <p>No recognition events found</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="px-4 py-3 text-left font-medium">Time</th>
                    <th className="px-4 py-3 text-left font-medium">Identity</th>
                    <th className="px-4 py-3 text-left font-medium">Source</th>
                    <th className="px-4 py-3 text-center font-medium">Status</th>
                    <th className="px-4 py-3 text-center font-medium">Trust</th>
                    <th className="px-4 py-3 text-center font-medium">Liveness</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((event) => (
                    <tr key={event.id} className="border-b last:border-0 hover:bg-muted/30">
                      <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                        {new Date(event.timestamp).toLocaleString()}
                      </td>
                      <td className="px-4 py-3">
                        {event.identity_name || (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-muted-foreground">
                        {event.source}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <Badge variant={event.is_known ? "success" : "secondary"}>
                          {event.is_known ? "Known" : "Unknown"}
                        </Badge>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className="font-mono text-xs">
                          {(event.trust_score * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <Badge variant={event.is_spoof ? "destructive" : "success"}>
                          {event.is_spoof ? "SPOOF" : "LIVE"}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

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
