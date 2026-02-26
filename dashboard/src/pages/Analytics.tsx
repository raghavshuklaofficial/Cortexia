/**
 * Analytics dashboard — overview stats, timeline, demographics.
 */

import { useEffect, useState } from "react";
import {
  BarChart3,
  Users,
  Shield,
  Eye,
  AlertTriangle,
  TrendingUp,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cortexiaApi, type TimelinePoint, type DemographicsData } from "@/lib/api";
import { useAppStore } from "@/lib/store";

export default function AnalyticsPage() {
  const { stats, setStats } = useAppStore();
  const [timeline, setTimeline] = useState<TimelinePoint[]>([]);
  const [demographics, setDemographics] = useState<DemographicsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const [overviewRes, timelineRes, demoRes] = await Promise.all([
          cortexiaApi.overview(),
          cortexiaApi.timeline(30, "day"),
          cortexiaApi.demographics(),
        ]);
        setStats(overviewRes);
        setTimeline(timelineRes);
        setDemographics(demoRes);
      } catch {
        // API not available
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [setStats]);

  const statCards = [
    {
      title: "Total Identities",
      value: stats?.total_identities ?? 0,
      icon: Users,
      color: "text-cortexia-400",
    },
    {
      title: "Known Events",
      value: stats?.known_events ?? 0,
      icon: Eye,
      color: "text-emerald-400",
    },
    {
      title: "Spoof Events",
      value: stats?.spoof_events ?? 0,
      icon: AlertTriangle,
      color: "text-red-400",
    },
    {
      title: "Avg Trust Score",
      value: stats ? `${(stats.avg_trust_score * 100).toFixed(0)}%` : "\u2014",
      icon: Shield,
      color: "text-amber-400",
    },
    {
      title: "Unknown Events",
      value: stats?.unknown_events ?? 0,
      icon: TrendingUp,
      color: "text-violet-400",
    },
    {
      title: "Total Events",
      value: stats?.total_events ?? 0,
      icon: BarChart3,
      color: "text-sky-400",
    },
  ];

  // Gender + age distribution from demographics
  const genderData = demographics?.gender_distribution
    ? Object.entries(demographics.gender_distribution).map(
        ([name, value]) => ({ name, value })
      )
    : [];

  const ageData = demographics?.age_distribution
    ? Object.entries(demographics.age_distribution).map(
        ([range, count]) => ({ range, count })
      )
    : [];

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center text-muted-foreground">
        Loading analytics...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Analytics</h2>
        <p className="text-sm text-muted-foreground">
          System-wide performance and recognition metrics
        </p>
      </div>

      {/* Stat Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        {statCards.map((stat) => (
          <Card key={stat.title}>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <stat.icon className={`h-8 w-8 ${stat.color} opacity-80`} />
                <div>
                  <p className="text-2xl font-bold">{stat.value}</p>
                  <p className="text-xs text-muted-foreground">{stat.title}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts */}
      <Tabs defaultValue="timeline">
        <TabsList>
          <TabsTrigger value="timeline">Recognition Timeline</TabsTrigger>
          <TabsTrigger value="demographics">Demographics</TabsTrigger>
        </TabsList>

        <TabsContent value="timeline">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                Recognition Events — Last 30 Days
              </CardTitle>
            </CardHeader>
            <CardContent>
              {timeline.length === 0 ? (
                <div className="flex h-64 items-center justify-center text-muted-foreground">
                  No data available
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={timeline}>
                    <defs>
                      <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#5c7cfa" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#5c7cfa" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                    <XAxis
                      dataKey="period"
                      tickFormatter={(d) =>
                        new Date(d).toLocaleDateString("en", {
                          month: "short",
                          day: "numeric",
                        })
                      }
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                    />
                    <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                        fontSize: "12px",
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="total"
                      stroke="#5c7cfa"
                      strokeWidth={2}
                      fillOpacity={1}
                      fill="url(#colorCount)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="demographics">
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Gender Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                {genderData.length === 0 ? (
                  <div className="flex h-48 items-center justify-center text-muted-foreground">
                    No data
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={genderData}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                      <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                      <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Bar dataKey="value" fill="#5c7cfa" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Age Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                {ageData.length === 0 ? (
                  <div className="flex h-48 items-center justify-center text-muted-foreground">
                    No data
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={ageData}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                      <XAxis dataKey="range" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                      <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Bar dataKey="count" fill="#748ffc" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
