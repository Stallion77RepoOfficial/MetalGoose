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

/// A slider over a few chosen values rather than a range, so a setting that can only be 2 or 4 has two
/// stops and not three.
struct SliderRow: View {
    let label: LocalizedStringKey
    @Binding var value: Int
    let values: [Int]

    /// The stop the binding stands for: the highest one it reaches, or the first. A stored value the
    /// current choices cannot reach never shows a figure the pipeline is not running at.
    private var index: Int {
        values.lastIndex { $0 <= value } ?? 0
    }

    var body: some View {
        HStack {
            Text(label).foregroundColor(.gray)
            Spacer()
            // One value would make the slider divide by its own zero width.
            if values.count > 1 {
                Slider(
                    value: Binding(
                        get: { Double(index) },
                        set: { value = values[min(values.count - 1, max(0, Int($0.rounded())))] }),
                    in: 0...Double(values.count - 1),
                    step: 1)
                .frame(minWidth: 110, maxWidth: 160)
            }
            Text(String(localized: "Frame multiplier", defaultValue: "\(values[index])×"))
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
