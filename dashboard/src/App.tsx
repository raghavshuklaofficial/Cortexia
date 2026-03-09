/**
 * Main App with routing.
 */

import { Routes, Route } from "react-router-dom";
import { Layout } from "./components/layout/Layout";
import LiveFeedPage from "./pages/LiveFeed";
import IdentitiesPage from "./pages/Identities";
import EventsPage from "./pages/Events";
import AnalyticsPage from "./pages/Analytics";
import ClustersPage from "./pages/Clusters";
import ForensicsPage from "./pages/Forensics";
import SettingsPage from "./pages/Settings";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<LiveFeedPage />} />
        <Route path="/identities" element={<IdentitiesPage />} />
        <Route path="/events" element={<EventsPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/clusters" element={<ClustersPage />} />
        <Route path="/forensics" element={<ForensicsPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}
