import SwiftUI

/// A titled card of settings.
struct ConfigPanel<Content: View>: View {
    let title: LocalizedStringKey
    let content: Content

    init(title: LocalizedStringKey, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text(title).font(.title3).bold()
            Divider().background(Color.gray)
            content
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.windowBackgroundColor))
        .cornerRadius(10)
    }
}

struct PickerRow<Option: SettingOption>: View {
    let label: LocalizedStringKey
    @Binding var selection: Option

    var body: some View {
        HStack {
            Text(label).foregroundColor(.gray)
            Spacer()
            Picker("", selection: $selection) {
                ForEach(Array(Option.allCases)) { option in
                    Text(option.title).tag(option)
                }
            }
            .labelsHidden()
            .frame(minWidth: 160, maxWidth: 220)
        }
    }
}

struct SliderRow: View {
    let label: LocalizedStringKey
    @Binding var value: Int
    let range: ClosedRange<Int>

    /// The displayed value is the binding clamped to the range, so a stored value the current range
    /// cannot reach never shows a figure the pipeline is not running at.
    private var clamped: Int {
        min(range.upperBound, max(range.lowerBound, value))
    }

    var body: some View {
        HStack {
            Text(label).foregroundColor(.gray)
            Spacer()
            // A range with a single value would make the slider divide by its own zero width.
            if range.lowerBound < range.upperBound {
                Slider(
                    value: Binding(
                        get: { Double(clamped) },
                        set: { value = min(range.upperBound, max(range.lowerBound, Int($0.rounded()))) }),
                    in: Double(range.lowerBound)...Double(range.upperBound),
                    step: 1)
                .frame(minWidth: 110, maxWidth: 160)
            }
            Text(verbatim: "\(clamped)x")
                .font(.system(.caption, design: .monospaced))
                .frame(width: 28, alignment: .trailing)
        }
    }
}

struct ToggleRow: View {
    let label: LocalizedStringKey
    @Binding var isOn: Bool

    var body: some View {
        HStack {
            Text(label).foregroundColor(.gray)
            Spacer()
            Toggle("", isOn: $isOn).labelsHidden()
        }
    }
}
