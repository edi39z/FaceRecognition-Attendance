"use client"

import { AnimatePresence, motion } from "framer-motion"
import { CheckCircle2, Clock, User, BadgeIcon as IdCard, Camera, RotateCcw, Sparkles, Calendar } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import type { AttendanceCardProps } from "@/lib/types"
import { format } from "date-fns"

export function AttendanceCard({ data, onReset }: AttendanceCardProps) {
  const getStatusColor = (message: string) => {
    if (message.toLowerCase().includes("tepat waktu") || message.toLowerCase().includes("hadir")) {
      return {
        bg: "from-emerald-500 to-green-600",
        text: "text-emerald-700 dark:text-emerald-300",
        bgLight: "bg-emerald-50 dark:bg-emerald-900/20",
        border: "border-emerald-200 dark:border-emerald-800",
        icon: "text-emerald-600 dark:text-emerald-400",
        pulse: "bg-emerald-500",
      }
    } else if (message.toLowerCase().includes("terlambat")) {
      return {
        bg: "from-orange-500 to-amber-600",
        text: "text-orange-700 dark:text-orange-300",
        bgLight: "bg-orange-50 dark:bg-orange-900/20",
        border: "border-orange-200 dark:border-orange-800",
        icon: "text-orange-600 dark:text-orange-400",
        pulse: "bg-orange-500",
      }
    }
    return {
      bg: "from-emerald-500 to-teal-600",
      text: "text-emerald-700 dark:text-emerald-300",
      bgLight: "bg-emerald-50 dark:bg-emerald-900/20",
      border: "border-emerald-200 dark:border-emerald-800",
      icon: "text-emerald-600 dark:text-emerald-400",
      pulse: "bg-emerald-500",
    }
  }

  const statusColors = getStatusColor(data.message || "")

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: -20 }}
        transition={{
          duration: 0.6,
          type: "spring",
          stiffness: 300,
          damping: 30,
        }}
        className="w-full max-w-2xl mx-auto"
      >
        <Card className="overflow-hidden border-2 border-emerald-100 dark:border-gray-700 shadow-2xl bg-white dark:bg-gray-800 rounded-3xl">
          {/* Success Header with Enhanced Gradient */}
          <CardHeader className="relative p-0">
            <div className={`bg-gradient-to-br ${statusColors.bg} p-10 text-white relative overflow-hidden`}>
              {/* Enhanced Background Pattern */}
              <div className="absolute inset-0 opacity-10">
                <div className="absolute top-6 right-6 animate-pulse">
                  <Sparkles className="h-10 w-10" />
                </div>
                <div className="absolute bottom-6 left-6 animate-pulse" style={{ animationDelay: "1s" }}>
                  <Sparkles className="h-8 w-8" />
                </div>
                <div className="absolute top-1/3 left-1/4 animate-pulse" style={{ animationDelay: "2s" }}>
                  <Sparkles className="h-6 w-6" />
                </div>
                <div className="absolute bottom-1/3 right-1/4 animate-pulse" style={{ animationDelay: "0.5s" }}>
                  <Sparkles className="h-7 w-7" />
                </div>
              </div>

              {/* Floating Orbs */}
              <div className="absolute inset-0 overflow-hidden">
                <div className="absolute top-10 left-10 w-20 h-20 bg-white/10 rounded-full blur-xl animate-pulse"></div>
                <div
                  className="absolute bottom-10 right-10 w-16 h-16 bg-white/10 rounded-full blur-xl animate-pulse"
                  style={{ animationDelay: "1.5s" }}
                ></div>
                <div
                  className="absolute top-1/2 right-20 w-12 h-12 bg-white/10 rounded-full blur-xl animate-pulse"
                  style={{ animationDelay: "0.8s" }}
                ></div>
              </div>

              <div className="relative z-10 text-center">
                <motion.div
                  initial={{ scale: 0, rotate: -180 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{ delay: 0.2, type: "spring", stiffness: 300, duration: 0.8 }}
                  className="inline-flex items-center justify-center w-24 h-24 bg-white/20 backdrop-blur-sm rounded-full mb-6 shadow-2xl"
                >
                  <CheckCircle2 className="h-12 w-12 text-white drop-shadow-lg" />
                </motion.div>

                <motion.h2
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                  className="text-3xl font-bold mb-3 drop-shadow-lg"
                >
                  Absensi Berhasil! ✨
                </motion.h2>

                <motion.p
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                  className="text-white/90 text-lg font-medium"
                >
                  Kehadiran Anda telah tercatat dalam sistem
                </motion.p>


              </div>
            </div>
          </CardHeader>

          <CardContent className="p-8 space-y-6">
            {/* Employee Information Grid */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.7 }}
              className="grid grid-cols-1 md:grid-cols-2 gap-4"
            >
              {/* Name */}
              <div className="flex items-center gap-4 p-5 bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-2xl border border-emerald-100 dark:border-emerald-800 hover:shadow-lg transition-all duration-300">
                <div className="w-14 h-14 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-2xl flex items-center justify-center shadow-lg">
                  <User className="h-7 w-7 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-emerald-600 dark:text-emerald-400 mb-1">Nama Karyawan</p>
                  <p className="text-lg font-bold text-gray-900 dark:text-white truncate">
                    {data.nama || "Tidak tersedia"}
                  </p>
                </div>
              </div>

              {/* NIP */}
              <div className="flex items-center gap-4 p-5 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-2xl border border-purple-100 dark:border-purple-800 hover:shadow-lg transition-all duration-300">
                <div className="w-14 h-14 bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center shadow-lg">
                  <IdCard className="h-7 w-7 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-purple-600 dark:text-purple-400 mb-1">NIM</p>
                  <p className="text-lg font-mono font-bold text-gray-900 dark:text-white truncate">
                    {data.nip || "Tidak tersedia"}
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Time and Date Information */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.8 }}
              className="grid grid-cols-1 md:grid-cols-2 gap-4"
            >
              {/* Date */}
              <div className="flex items-center gap-4 p-5 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-2xl border border-blue-100 dark:border-blue-800 hover:shadow-lg transition-all duration-300">
                <div className="w-14 h-14 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg">
                  <Calendar className="h-7 w-7 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-blue-600 dark:text-blue-400 mb-1">Tanggal</p>
                  <p className="text-lg font-bold text-gray-900 dark:text-white">
                    {data.timestamp ? format(new Date(data.timestamp), "dd MMMM yyyy") : "Tidak tersedia"}
                  </p>
                </div>
              </div>

              {/* Time */}
              <div className="flex items-center gap-4 p-5 bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 rounded-2xl border border-orange-100 dark:border-orange-800 hover:shadow-lg transition-all duration-300">
                <div className="w-14 h-14 bg-gradient-to-r from-orange-500 to-amber-600 rounded-2xl flex items-center justify-center shadow-lg">
                  <Clock className="h-7 w-7 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-orange-600 dark:text-orange-400 mb-1">Waktu Absensi</p>
                  <p className="text-lg font-bold text-gray-900 dark:text-white font-mono">
                    {data.timestamp ? format(new Date(data.timestamp), "HH:mm:ss") : "Tidak tersedia"}
                  </p>
                </div>
              </div>
            </motion.div>


            {/* Enhanced Status Message */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.0 }}
              className={`p-6 ${statusColors.bgLight} ${statusColors.border} border-2 rounded-3xl relative overflow-hidden`}
            >
              <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-white/10 to-transparent rounded-full -translate-y-16 translate-x-16"></div>
              <div className="relative flex items-center gap-4">
                <div
                  className={`w-12 h-12 bg-gradient-to-r ${statusColors.bg} rounded-2xl flex items-center justify-center shadow-lg`}
                >
                  <CheckCircle2 className="h-6 w-6 text-white" />
                </div>
                <div className="flex-1">
                  <p className="font-bold text-gray-900 dark:text-white text-lg mb-1">Status Kehadiran</p>
                  <p className={`${statusColors.text} font-semibold text-lg`}>
                    {data.message || "Absensi berhasil dicatat"}
                  </p>
                </div>
              </div>
            </motion.div>
          </CardContent>

          <CardFooter className="p-8 pt-0">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2 }}
              className="w-full"
            >
              <Button
                onClick={onReset}
                className="w-full h-16 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-bold text-lg rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-[1.02] relative overflow-hidden group"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-emerald-400 to-teal-400 opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
                <div className="relative flex items-center justify-center gap-3">
                  <CheckCircle2 className="h-6 w-6" />
                  <span>Tandai Absensi</span>
                </div>
              </Button>
            </motion.div>
          </CardFooter>
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}
